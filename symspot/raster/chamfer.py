"""Chamfer-поиск эталона по растру. Фаза 5.

Схема (20 эталонов на листе 7200x4800 за около 40 с, то есть порядка 2 с на эталон):

1. Плотный грубый проход на сильно уменьшенном растре (диагональ эталона ``raster_dense_diag``
   px): для каждой позы (зеркало, угол с шагом 2 * ``raster_angle_step``, несколько масштабов)
   средний усеченный chamfer шаблона считается сразу для всех центров суммой сдвинутых карт
   расстояний, с добавкой штрафа за избыток туши вокруг центра (плотные места). Берутся
   локальные минимумы минимума по позам.
2. На растре с диагональю эталона ``raster_pyr_diag`` px в этих центрах перебираются позы
   (угол и масштаб с удвоенным шагом); значение = прямой chamfer + приближение обратного
   (доля туши в квадрате экземпляра, не объясненная шаблоном).
3. Лучшие позы уточняются покоординатным спуском по углу, масштабу и центру (шаги
   ``raster_angle_step``, ``raster_scale_step``, до 3 px, на последних шагах 1 и 0.5 px).
4. Порог ``raster_chamfer_max``, NMS по центру и углу, точный обратный chamfer (медиана
   расстояний от туши круга экземпляра до шаблона) для ранжирования, топ ``raster_topk``.

Прямой chamfer (точки шаблона -> тушь) мал в любом плотном месте, поэтому без обратного
члена гипотезы ранжируются плохо.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.types import Hypothesis, Query, Similarity
from symspot.raster.preprocess import RasterPrep

# Радиус NMS итоговых гипотез по центру (доли диагонали).
CENTER_NMS = 0.25
# Разность углов (градусы), меньше которой гипотезы с близкими центрами считаются дублями.
NMS_ANGLE_DEG = 25.0
# Число масштабов плотного прохода.
DENSE_SCALES = 4
# Радиус локального минимума плотного прохода (доли диагонали).
DENSE_NMS = 0.12
# Порог плотного прохода = DENSE_FACTOR * ``raster_chamfer_max``.
DENSE_FACTOR = 2.5
# Подавление соседних центров кандидатов (доли диагонали).
CAND_NMS = 0.12
# Штраф плотных мест: доля туши круга экземпляра относительно туши эталона, не штрафуемая (запас),
# и верхний предел штрафуемой доли.
DENSE_SLACK = 1.5
DENSE_EXCESS_CAP = 4.0
# Расстояние (px уровня), не больше которого точка шаблона считается попавшей на тушь,
# и полуразмер квадрата экземпляра (доли диагонали) в приближении обратного chamfer.
HIT_TOL = 1.0
BOX_HALF = 0.39
# Число лучших поз на центр после перебора поз и разность углов между ними (градусы).
POSES_PER_CENTER = 3
SECOND_ANGLE_GAP_DEG = 40.0
# Минимальное число пикселей скелета в ячейке, чтобы ячейка считалась занятой.
OCC_MIN_PX = 0.4
# Усечение обратного chamfer (доля размера экземпляра).
REVERSE_TRUNC = 0.2
# Шаги спуска при уточнении: сдвиг центра (px уменьшенного растра), множитель шага угла/масштаба
# и признак слабого усечения расстояния (шире область притяжения на первых шагах).
REFINE_STEPS = (
    (3.0, 1.0, True),
    (2.0, 1.0, True),
    (1.5, 0.5, True),
    (1.0, 1.0, False),
    (0.5, 0.5, False),
)
# Кандидатов в пачке при уточнении (ограничение памяти).
REFINE_CHUNK = 1000
# Во сколько раз слабее усечение расстояния на первых шагах уточнения.
REFINE_SOFT_TRUNC = 3.0


@dataclass
class _Level:
    """Уменьшенный растр: занятые ячейки и расстояние до них."""

    fx: float
    fy: float
    f: float  # средний коэффициент уменьшения
    occ: np.ndarray  # bool, занятые ячейки
    dist: np.ndarray  # float32, расстояние до занятых ячеек (px уменьшенного растра)
    count: np.ndarray  # float32, пикселей скелета исходного растра в ячейке


def _build_level(prep: RasterPrep, f: float) -> _Level:
    h, w = prep.skeleton.shape
    ws, hs = max(1, round(w / f)), max(1, round(h / f))
    fx, fy = w / ws, h / hs
    sk = prep.skeleton.view(np.uint8).astype(np.float32)
    count = sk if (ws == w and hs == h) else cv2.resize(sk, (ws, hs), interpolation=cv2.INTER_AREA)
    count = count * np.float32(fx * fy)
    occ = count > OCC_MIN_PX
    dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5)
    return _Level(fx, fy, 0.5 * (fx + fy), occ, dist.astype(np.float32), count)


def _template_points(query: Query, n: int) -> np.ndarray:
    """Равномерно прореженные точки шаблона (нормализованные координаты)."""
    pts = np.asarray(query.samples, dtype=np.float64)
    if len(pts) > n:
        pts = pts[np.unique(np.linspace(0, len(pts) - 1, n).astype(np.int64))]
    return pts


def _lookup(dist: np.ndarray, x: np.ndarray, y: np.ndarray, trunc: float) -> np.ndarray:
    """Билинейное значение карты расстояний в непрерывных координатах (центр пикселя i = i + 0.5);
    за пределами растра = ``trunc``. Массивы любой формы."""
    shape = x.shape
    mx = (x.reshape(-1) - 0.5).astype(np.float32)
    my = (y.reshape(-1) - 0.5).astype(np.float32)
    n = mx.size
    cols = 4096
    rows = -(-n // cols)
    pad = rows * cols - n
    if pad:
        mx = np.concatenate([mx, np.zeros(pad, np.float32)])
        my = np.concatenate([my, np.zeros(pad, np.float32)])
    out = cv2.remap(
        dist,
        mx.reshape(rows, cols),
        my.reshape(rows, cols),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(trunc),
    )
    return out.reshape(-1)[:n].reshape(shape)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rotate(pts: np.ndarray, theta: float, mirror: bool) -> np.ndarray:
    """p' = R(theta) M p для точек (n, 2) (как ``Similarity`` с s = 1)."""
    c, s = math.cos(theta), math.sin(theta)
    x = pts[:, 0]
    y = -pts[:, 1] if mirror else pts[:, 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=1)


def _query_ink(query: Query, prep: RasterPrep, cfg: Config) -> float:
    """Пикселей скелета исходного растра в рамке эталона (оценка по сэмплам, если рамки нет)."""
    if query.box is not None:
        c = query.frame.apply(
            np.array([[query.box[0], query.box[1]], [query.box[2], query.box[3]]])
        )
        h, w = prep.skeleton.shape
        x0, y0 = max(0, int(math.floor(c[0, 0]))), max(0, int(math.floor(c[0, 1])))
        x1, y1 = min(w, int(math.ceil(c[1, 0]))), min(h, int(math.ceil(c[1, 1])))
        n = int(prep.skeleton[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
        if n > 0:
            return float(n)
    return float(len(query.samples)) * max(1.0, cfg.sample_step * query.diag)


def _disc_kernel(radius: float) -> np.ndarray:
    """Круглое ядро (площадь пикселя считается по подвыборке 4x4), сумма значений = число ячеек."""
    r = int(math.ceil(radius)) + 1
    g = (np.arange((2 * r + 1) * 4) + 0.5) / 4 - r - 0.5
    yy, xx = np.meshgrid(g, g, indexing="ij")
    m = (np.hypot(xx, yy) <= radius).astype(np.float32)
    return m.reshape(2 * r + 1, 4, 2 * r + 1, 4).mean(axis=(1, 3))


def _query_cells(query: Query, lvl: _Level) -> float:
    """Занятых ячеек уровня в рамке эталона на листе (длина эталона в ячейках при масштабе 1)."""
    if query.box is not None:
        c = query.frame.apply(
            np.array([[query.box[0], query.box[1]], [query.box[2], query.box[3]]])
        )
        h, w = lvl.occ.shape
        x0, y0 = (
            max(0, int(math.floor(c[0, 0] / lvl.fx))),
            max(0, int(math.floor(c[0, 1] / lvl.fy))),
        )
        x1, y1 = min(w, int(math.ceil(c[1, 0] / lvl.fx))), min(h, int(math.ceil(c[1, 1] / lvl.fy)))
        if x1 > x0 and y1 > y0:
            n = int(lvl.occ[y0:y1, x0:x1].sum())
            if n > 0:
                return float(n)
    return float(len(query.samples)) * max(1.0, 1.0 / lvl.f)


def _box_sum(ii: np.ndarray, cx: np.ndarray, cy: np.ndarray, half: float) -> np.ndarray:
    """Сумма по квадратам с центрами (cx, cy) и полуразмером ``half`` по интегральному
    изображению ``ii`` (форма (h + 1, w + 1))."""
    h, w = ii.shape[0] - 1, ii.shape[1] - 1
    x0 = np.clip(np.floor(cx - half).astype(np.int64), 0, w)
    x1 = np.clip(np.ceil(cx + half).astype(np.int64), 0, w)
    y0 = np.clip(np.floor(cy - half).astype(np.int64), 0, h)
    y1 = np.clip(np.ceil(cy + half).astype(np.int64), 0, h)
    return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]).astype(np.float64)


@dataclass
class Candidates:
    """Кандидаты (координаты в px уменьшенного растра ``lvl``)."""

    lvl: _Level
    diag_s: float  # диагональ эталона в уменьшенном растре, px
    pts: np.ndarray  # точки шаблона, нормализованные координаты
    frac: np.ndarray  # chamfer, доли диагонали экземпляра
    cx: np.ndarray
    cy: np.ndarray
    s: np.ndarray
    theta: np.ndarray
    mirror: np.ndarray  # bool

    def __len__(self) -> int:
        return len(self.frac)


def _dense_pass(
    query: Query, prep: RasterPrep, cfg: Config, mirrors: list[bool], a_step: float
) -> tuple[_Level, np.ndarray] | None:
    """Плотный грубый проход. Возвращает уровень и центры (N, 2) в px уровня: локальные
    минимумы лучшего по позам chamfer с добавкой штрафа плотных мест (по возрастанию)."""
    f = max(1.0, query.diag / cfg.raster_dense_diag)
    lvl = _build_level(prep, f)
    diag_d = query.diag / lvl.f
    pts = _template_points(query, cfg.raster_dense_pts)
    trunc = cfg.raster_trunc * diag_d * cfg.scale_min
    dist = np.minimum(lvl.dist, np.float32(trunc))
    h, w = dist.shape
    scales = np.geomspace(cfg.scale_min, cfg.scale_max, DENSE_SCALES)
    margin = int(math.ceil(0.5 * scales[-1] * diag_d)) + 2
    padded = np.pad(dist, margin, constant_values=trunc)
    n_ang = max(4, round(2 * math.pi / (2 * a_step)))
    angles = np.arange(n_ang) * (2 * math.pi / n_ang)
    # штраф за избыток туши в круге экземпляра (плотные места): доля туши круга сверх эталона
    ink = _query_ink(query, prep, cfg)
    penalty = []
    for s in scales:
        disc = cv2.filter2D(
            lvl.count, -1, _disc_kernel(0.5 * s * diag_d), borderType=cv2.BORDER_CONSTANT
        )
        ratio = disc / np.float32(max(ink * s, 1.0))
        penalty.append(
            (cfg.raster_dense_excess * np.clip(ratio - DENSE_SLACK, 0.0, DENSE_EXCESS_CAP)).astype(
                np.float32
            )
        )
    best = np.full((h, w), np.inf, dtype=np.float32)
    acc = np.empty((h, w), dtype=np.float32)
    n = len(pts)
    for mirror in mirrors:
        for th in angles:
            base = _rotate(pts, float(th), mirror) * diag_d
            for si, s in enumerate(scales):
                off = np.rint(base * s).astype(np.int64)
                acc[:] = 0.0
                for dx, dy in off:
                    acc += padded[margin + dy : margin + dy + h, margin + dx : margin + dx + w]
                acc *= np.float32(1.0 / (n * s * diag_d))
                acc += penalty[si]
                np.minimum(best, acc, out=best)
    r = max(1, round(DENSE_NMS * diag_d))
    local_min = cv2.erode(
        best,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=1e9,
    )
    ys, xs = np.nonzero((best <= local_min) & (best < DENSE_FACTOR * cfg.raster_chamfer_max))
    if len(xs) == 0:
        return None
    order = np.argsort(best[ys, xs], kind="stable")[: cfg.raster_cand_k]
    ys, xs = ys[order], xs[order]
    return lvl, np.stack([xs + 0.5, ys + 0.5], axis=1)


def _dedupe_centers(centers: np.ndarray, radius: float, limit: int) -> np.ndarray:
    """Индексы центров (уже по возрастанию значения) после подавления соседей в ``radius``."""
    neighbors = cKDTree(centers).query_ball_point(centers, r=radius)
    alive = np.ones(len(centers), dtype=bool)
    keep: list[int] = []
    for i in range(len(centers)):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) >= limit:
            break
        alive[neighbors[i]] = False
    return np.array(keep, dtype=np.int64)


def _pose_pass(
    centers: np.ndarray,
    lvl: _Level,
    pts: np.ndarray,
    diag_s: float,
    cfg: Config,
    mirrors: list[bool],
    a_step: float,
    cells: float,
) -> Candidates | None:
    """Перебор поз (зеркало, угол с шагом 2 * шаг, масштаб с шагом 2 * шаг) в центрах на
    растре уточнения; по ``POSES_PER_CENTER`` лучших поз (разные углы/зеркало) на центр.

    Значение позы = прямой chamfer + ``raster_rev_weight`` * приближение обратного chamfer:
    доля занятых ячеек в квадрате экземпляра, не объясненная шаблоном
    (1 - ``cells`` * масштаб * доля точек шаблона на туши / число занятых ячеек квадрата).
    ``cells`` — число занятых ячеек эталона при масштабе 1."""
    trunc = cfg.raster_trunc * diag_s * cfg.scale_min
    dist = np.minimum(lvl.dist, np.float32(trunc))
    n_ang = max(4, round(2 * math.pi / (2 * a_step)))
    angles = np.arange(n_ang) * (2 * math.pi / n_ang)
    scales = np.arange(cfg.scale_min, cfg.scale_max + 1e-9, 2 * cfg.raster_scale_step)
    if len(scales) == 0:
        scales = np.array([1.0])
    ii = cv2.integral(lvl.occ.astype(np.uint8), sdepth=cv2.CV_32S)
    nc, n_pts = len(centers), len(pts)
    best = np.full((len(mirrors), nc, n_ang), np.inf, dtype=np.float32)
    best_s = np.zeros((len(mirrors), nc, n_ang), dtype=np.int16)
    chunk = max(1, 400_000 // (n_ang * n_pts))
    for mi, mirror in enumerate(mirrors):
        tmpl = np.stack([_rotate(pts, float(a), mirror) for a in angles]) * diag_s
        for si, s in enumerate(scales):
            for c0 in range(0, nc, chunk):
                cc = centers[c0 : c0 + chunk]
                x = cc[:, 0, None, None] + s * tmpl[None, :, :, 0]
                y = cc[:, 1, None, None] + s * tmpl[None, :, :, 1]
                dl = _lookup(dist, x, y, trunc)
                val = dl.mean(axis=2) / (s * diag_s)
                hit = (dl <= HIT_TOL).mean(axis=2)  # (nch, n_ang)
                area = _box_sum(ii, cc[:, 0], cc[:, 1], BOX_HALF * s * diag_s)  # (nch,)
                unexplained = np.clip(1.0 - cells * s * hit / np.maximum(area[:, None], 1.0), 0, 1)
                val += (cfg.raster_rev_weight * unexplained).astype(np.float32)
                cur = best[mi, c0 : c0 + chunk]
                better = val < cur
                cur[better] = val[better]
                best_s[mi, c0 : c0 + chunk][better] = si
    flat = best.transpose(1, 0, 2).reshape(nc, -1)  # (nc, M * n_ang)
    gap = max(1, round(math.radians(SECOND_ANGLE_GAP_DEG) / (2 * math.pi / n_ang)))
    thr = DENSE_FACTOR * cfg.raster_chamfer_max
    rows = np.arange(nc)
    work = flat.copy()
    parts_c, parts_i, parts_v = [], [], []
    for _ in range(POSES_PER_CENTER):
        i1 = work.argmin(axis=1)
        parts_c.append(rows)
        parts_i.append(i1)
        parts_v.append(work[rows, i1])
        m1, a1 = np.divmod(i1, n_ang)
        for d in range(-gap, gap + 1):
            work[rows, m1 * n_ang + (a1 + d) % n_ang] = np.inf
    cidx, idx, val = np.concatenate(parts_c), np.concatenate(parts_i), np.concatenate(parts_v)
    keep = val <= thr
    cidx, idx, val = cidx[keep], idx[keep], val[keep]
    if len(val) == 0:
        return None
    mi_e, ai_e = np.divmod(idx, n_ang)
    si_e = best_s[mi_e, cidx, ai_e]
    return Candidates(
        lvl=lvl,
        diag_s=diag_s,
        pts=pts,
        frac=val.astype(float),
        cx=centers[cidx, 0],
        cy=centers[cidx, 1],
        s=scales[si_e],
        theta=angles[ai_e],
        mirror=np.array(mirrors)[mi_e],
    )


def reverse_scores(cand: Candidates) -> np.ndarray:
    """Обратный chamfer: медианное (усеченное) расстояние от занятых ячеек растра внутри
    описанного круга экземпляра до ближайшей точки шаблона, доли размера экземпляра.

    Прямой chamfer (точки шаблона -> тушь) мал в любом плотном месте; обратный отсеивает его.
    Медиана устойчива к линиям, проходящим рядом с символом.
    """
    occ = cand.lvl.occ
    h, w = occ.shape
    out = np.ones(len(cand))
    for i in range(len(cand)):
        size = float(cand.s[i]) * cand.diag_s
        radius = 0.5 * size
        x0, x1 = int(max(0, cand.cx[i] - radius - 1)), int(min(w, cand.cx[i] + radius + 2))
        y0, y1 = int(max(0, cand.cy[i] - radius - 1)), int(min(h, cand.cy[i] + radius + 2))
        ys, xs = np.nonzero(occ[y0:y1, x0:x1])
        if len(xs) == 0:
            continue
        px, py = xs + x0 + 0.5 - cand.cx[i], ys + y0 + 0.5 - cand.cy[i]
        inside = np.hypot(px, py) <= radius
        if not inside.any():
            continue
        tpl = _rotate(cand.pts, float(cand.theta[i]), bool(cand.mirror[i])) * size
        d = np.hypot(px[inside, None] - tpl[None, :, 0], py[inside, None] - tpl[None, :, 1])
        out[i] = float(np.median(np.minimum(d.min(axis=1), REVERSE_TRUNC * size))) / size
    return out


def _refine(c: Candidates, cfg: Config, a_step: float) -> Candidates | None:
    """Покоординатный спуск по углу, масштабу и центру (пачками по ``REFINE_CHUNK``)."""
    parts = [
        _refine_chunk(coarse_subset(c, np.arange(i, min(len(c), i + REFINE_CHUNK))), cfg, a_step)
        for i in range(0, len(c), REFINE_CHUNK)
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return Candidates(
        lvl=c.lvl,
        diag_s=c.diag_s,
        pts=c.pts,
        frac=np.concatenate([p.frac for p in parts]),
        cx=np.concatenate([p.cx for p in parts]),
        cy=np.concatenate([p.cy for p in parts]),
        s=np.concatenate([p.s for p in parts]),
        theta=np.concatenate([p.theta for p in parts]),
        mirror=np.concatenate([p.mirror for p in parts]),
    )


def _refine_chunk(c: Candidates, cfg: Config, a_step: float) -> Candidates | None:
    lvl, diag_s, pts = c.lvl, c.diag_s, c.pts
    trunc = cfg.raster_trunc * diag_s * cfg.scale_min
    dist = np.minimum(lvl.dist, np.float32(trunc))
    dist_soft = np.minimum(lvl.dist, np.float32(REFINE_SOFT_TRUNC * trunc))
    ne = len(c)
    cx, cy, th, sc = c.cx.copy(), c.cy.copy(), c.theta.copy(), c.s.copy()
    sgn = np.where(c.mirror, -1.0, 1.0)  # зеркало: y -> -y до поворота
    s_lo, s_hi = 0.5 * cfg.scale_min, 1.5 * cfg.scale_max
    cur = np.full(ne, np.inf)
    was_soft = True
    e = np.arange(ne)
    for step_pos, step_mul, soft in REFINE_STEPS:
        if was_soft and not soft:
            cur = np.full(ne, np.inf)  # значения при другом усечении несравнимы
        was_soft = soft
        da = np.array([-a_step, 0.0, a_step]) * step_mul
        ds = np.array([-cfg.raster_scale_step, 0.0, cfg.raster_scale_step]) * step_mul
        offs = np.array([-1.0, 0.0, 1.0]) * step_pos
        ox, oy = (v.ravel() for v in np.meshgrid(offs, offs))
        angs = th[:, None] + da[None, :]  # (E, 3)
        scs = np.clip(sc[:, None] + ds[None, :], s_lo, s_hi)
        co, si = np.cos(angs), np.sin(angs)
        xm = pts[None, None, :, 0]
        ym = sgn[:, None, None] * pts[None, None, :, 1]
        tx = (co[:, :, None] * xm - si[:, :, None] * ym) * diag_s  # (E, 3, n)
        ty = (si[:, :, None] * xm + co[:, :, None] * ym) * diag_s
        # оси: кандидат (E), масштаб (3), угол (3), смещение (9), точки (n)
        x = (
            cx[:, None, None, None, None]
            + ox[None, None, None, :, None]
            + scs[:, :, None, None, None] * tx[:, None, :, None, :]
        )
        y = (
            cy[:, None, None, None, None]
            + oy[None, None, None, :, None]
            + scs[:, :, None, None, None] * ty[:, None, :, None, :]
        )
        d_map, d_cap = (dist_soft, REFINE_SOFT_TRUNC * trunc) if soft else (dist, trunc)
        fr = _lookup(d_map, x, y, d_cap).mean(axis=4) / (scs[:, :, None, None] * diag_s)
        flat = fr.reshape(ne, -1)
        bi = flat.argmin(axis=1)
        bv = flat[e, bi]
        si_b, ai_b, oi_b = np.unravel_index(bi, (3, 3, 9))
        better = bv < cur
        cx = np.where(better, cx + ox[oi_b], cx)
        cy = np.where(better, cy + oy[oi_b], cy)
        th = np.where(better, angs[e, ai_b], th)
        sc = np.where(better, scs[e, si_b], sc)
        cur = np.where(better, bv, cur)
    ok = np.nonzero(cur < cfg.raster_chamfer_max)[0]
    if len(ok) == 0:
        return None
    return Candidates(
        lvl=lvl,
        diag_s=diag_s,
        pts=pts,
        frac=cur[ok].astype(float),
        cx=cx[ok],
        cy=cy[ok],
        s=sc[ok],
        theta=th[ok],
        mirror=c.mirror[ok],
    )


def find_candidates(query: Query, prep: RasterPrep, cfg: Config) -> Candidates | None:
    """Плотный грубый проход -> позы в центрах на растре уточнения -> отбор по сумме прямого и
    обратного chamfer -> спуск. Кандидаты с chamfer ниже ``raster_chamfer_max``
    (координаты в px уменьшенного растра)."""
    if len(query.samples) < 3 or query.diag <= 0 or not prep.skeleton.any():
        return None
    mirrors = [False, True] if cfg.allow_mirror else [False]
    a_step = math.radians(cfg.raster_angle_step)
    dense = _dense_pass(query, prep, cfg, mirrors, a_step)
    if dense is None:
        return None
    lvl_d, centers_d = dense
    keep = _dedupe_centers(centers_d, CAND_NMS * query.diag / lvl_d.f, cfg.raster_cand_k)
    centers_d = centers_d[keep]

    f = max(1.0, query.diag / cfg.raster_pyr_diag)
    lvl = _build_level(prep, f)
    diag_s = query.diag / lvl.f
    pts = _template_points(query, cfg.raster_template_pts)
    centers = centers_d * np.array([lvl_d.fx / lvl.fx, lvl_d.fy / lvl.fy])
    coarse = _pose_pass(centers, lvl, pts, diag_s, cfg, mirrors, a_step, _query_cells(query, lvl))
    if coarse is None:
        return None
    # спуск дешев: уточняются все лучшие позы, а не только прошедшие обратный chamfer
    top = np.argsort(coarse.frac, kind="stable")[: cfg.raster_refine_top]
    return _refine(coarse_subset(coarse, top), cfg, a_step)


def coarse_subset(c: Candidates, idx: np.ndarray) -> Candidates:
    return Candidates(
        lvl=c.lvl,
        diag_s=c.diag_s,
        pts=c.pts,
        frac=c.frac[idx],
        cx=c.cx[idx],
        cy=c.cy[idx],
        s=c.s[idx],
        theta=c.theta[idx],
        mirror=c.mirror[idx],
    )


def _nms(cand: Candidates, order: np.ndarray, limit: int) -> list[int]:
    """NMS по центру и углу (то же зеркало) в порядке ``order``; не более ``limit`` индексов."""
    arr = np.stack([cand.cx, cand.cy, cand.theta], axis=1)
    alive = np.ones(len(cand), dtype=bool)
    rad = CENTER_NMS * cand.diag_s
    ang_thr = math.radians(NMS_ANGLE_DEG)
    keep: list[int] = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(int(i))
        if len(keep) >= limit:
            break
        d = np.hypot(arr[:, 0] - arr[i, 0], arr[:, 1] - arr[i, 1])
        da = np.abs(_wrap(arr[:, 2] - arr[i, 2]))
        alive &= ~((d < rad) & (da < ang_thr) & (cand.mirror == cand.mirror[i]))
    return keep


def chamfer_search(query: Query, prep: RasterPrep, cfg: Config) -> list[Hypothesis]:
    """Гипотезы T (нормализованный эталон -> пиксели исходного изображения).

    Углы с шагом ``raster_angle_step`` (при ``allow_mirror`` и с отражением), масштабы
    ``scale_min..scale_max`` с шагом ``raster_scale_step``; средний усеченный chamfer шаблона из
    точек ``query.samples`` ниже ``raster_chamfer_max`` * диагональ экземпляра -> гипотезы
    (``votes`` = -(chamfer + обратный chamfer) в долях диагонали, ``source="chamfer"``),
    NMS по центру и углу, топ ``raster_topk``.
    """
    cand = find_candidates(query, prep, cfg)
    if cand is None:
        return []
    lvl = cand.lvl
    keep = np.array(_nms(cand, np.argsort(cand.frac, kind="stable"), cfg.raster_rev_top))
    cand = coarse_subset(cand, keep)
    score = cand.frac + reverse_scores(cand)
    out: list[Hypothesis] = []
    for i in np.argsort(score, kind="stable")[: cfg.raster_topk]:
        out.append(
            Hypothesis(
                legend_id=query.legend_id,
                T=Similarity(
                    float(cand.s[i]) * query.diag,
                    float(_wrap(np.asarray(cand.theta[i]))),
                    float(cand.cx[i]) * lvl.fx,
                    float(cand.cy[i]) * lvl.fy,
                    bool(cand.mirror[i]),
                ),
                votes=-float(score[i]),
                source="chamfer",
                variant=query.variant,
            )
        )
    return out
