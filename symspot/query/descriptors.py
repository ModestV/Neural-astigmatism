"""3.2 Якоря примитивов и ключи пар, инвариантные к подобию. Ядро (Opus)."""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from symspot.core.config import Config
from symspot.core.geometry_utils import normalize_text
from symspot.core.types import Arc, Primitive, Segment, TextPrim

KIND_SEGMENT = 0
KIND_ARC = 1
KIND_CIRCLE = 2
KIND_TEXT = 3

# Якоря считаются совпадающими (концентричная пара), если расстояние между ними меньше этой
# доли большего из размеров. Это геометрическое определение вида ключа, а не порог качества.
CONCENTRIC_FRAC = 0.05
_TWO_PI = 2.0 * math.pi
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass
class Anchors:
    """Якоря примитивов (P штук).

    ``kind`` (P,) — KIND_*; ``pos`` (P, 2) — середина отрезка / центр дуги / центр рамки
    текста; ``dir`` (P,) — направление отрезка (по модулю π), направление на середину дуги
    от центра (ориентированное), угол базовой линии текста; для окружности NaN;
    ``size`` (P,) — длина отрезка / радиус / диагональ рамки текста;
    ``extent`` (P,) — угловой размах дуги (0 для прочих);
    ``text_hash`` (P,) — хеш нормализованной строки текста (0 для прочих).
    """

    kind: np.ndarray
    pos: np.ndarray
    dir: np.ndarray
    size: np.ndarray
    extent: np.ndarray
    text_hash: np.ndarray

    def __len__(self) -> int:
        return int(len(self.kind))


def text_hash(s: str) -> int:
    """Стабильный (не зависящий от PYTHONHASHSEED) хеш нормализованной строки, > 0."""
    return zlib.crc32(normalize_text(s).encode("utf-8")) + 1


def compute_anchors(prims: Sequence[Primitive]) -> Anchors:
    """Якоря для списка примитивов."""
    n = len(prims)
    kind = np.zeros(n, dtype=np.int64)
    pos = np.zeros((n, 2))
    dirs = np.full(n, np.nan)
    size = np.zeros(n)
    extent = np.zeros(n)
    th = np.zeros(n, dtype=np.int64)
    for k, p in enumerate(prims):
        if isinstance(p, Segment):
            kind[k] = KIND_SEGMENT
            pos[k] = ((p.p0[0] + p.p1[0]) / 2, (p.p0[1] + p.p1[1]) / 2)
            dx, dy = p.p1[0] - p.p0[0], p.p1[1] - p.p0[1]
            dirs[k] = math.atan2(dy, dx) % math.pi
            size[k] = math.hypot(dx, dy)
        elif isinstance(p, Arc):
            pos[k] = p.center
            size[k] = p.r
            if p.is_full:
                kind[k] = KIND_CIRCLE
            else:
                kind[k] = KIND_ARC
                dirs[k] = ((p.t0 + p.t1) / 2) % _TWO_PI
                extent[k] = p.t1 - p.t0
        elif isinstance(p, TextPrim):
            kind[k] = KIND_TEXT
            x0, y0, x1, y1 = p.bbox
            pos[k] = ((x0 + x1) / 2, (y0 + y1) / 2)
            dirs[k] = p.angle % _TWO_PI
            size[k] = math.hypot(x1 - x0, y1 - y0)
            th[k] = text_hash(p.text)
        else:
            raise TypeError(f"неизвестный примитив {type(p)}")
    return Anchors(kind, pos, dirs, size, extent, th)


def _period(kind: np.ndarray) -> np.ndarray:
    """Период направления: π для отрезка, 2π для дуги и текста."""
    return np.where(kind == KIND_SEGMENT, math.pi, _TWO_PI)


def _bins(value: np.ndarray, width: float, spread: bool) -> tuple[np.ndarray, np.ndarray]:
    """Номер ячейки и номер соседней (ближайшей к значению) ячейки."""
    x = value / width
    b0 = np.floor(x)
    if not spread:
        return b0.astype(np.int64), b0.astype(np.int64)
    frac = x - b0
    b1 = np.where(frac >= 0.5, b0 + 1, b0 - 1)
    return b0.astype(np.int64), b1.astype(np.int64)


def _wrap_bins(b: np.ndarray, nbins: np.ndarray) -> np.ndarray:
    return np.mod(b, nbins)


def _mix(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Перемешивание 64-битного хеша (вариант splitmix)."""
    with np.errstate(over="ignore"):
        h = (h ^ (v.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        h = h ^ (h >> np.uint64(29))
    return h & _MASK


def pair_keys(
    anchors: Anchors, i: np.ndarray, j: np.ndarray, cfg: Config, spread: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Ключи упорядоченных пар (i[k], j[k]), инвариантные к подобию (без отражения).

    Возвращает (keys int64, row), где row[m] — номер пары k, к которой относится keys[m].
    При ``spread=False`` ровно один ключ на пару. При ``spread=True`` (сторона эталона)
    к каждой паре добавляются ключи соседних ячеек квантования, чтобы значения у границы
    ячейки не терялись. Пары, для которых ключ не определен (обе окружности с общим центром,
    вырожденные), пропускаются.
    """
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if len(i) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ki, kj = anchors.kind[i], anchors.kind[j]
    pi, pj = anchors.pos[i], anchors.pos[j]
    si, sj = anchors.size[i], anchors.size[j]
    di, dj = anchors.dir[i], anchors.dir[j]
    dvec = pj - pi
    d = np.hypot(dvec[:, 0], dvec[:, 1])
    big = np.maximum(si, sj)
    conc = d < CONCENTRIC_FRAC * np.maximum(big, 1e-12)
    valid = (si > 0) & (sj > 0) & ~((ki == KIND_TEXT) & (kj == KIND_TEXT) & conc)
    # обе без направления (окружности) и концентричны — подобие не определено
    valid &= ~(conc & np.isnan(di) & np.isnan(dj))
    # концентричная пара с текстом не дает масштаба
    valid &= ~(conc & ((ki == KIND_TEXT) | (kj == KIND_TEXT)))

    ang_w = math.radians(cfg.key_angle_bin_deg)
    rat_w = cfg.ratio_bin_log
    phi = np.arctan2(dvec[:, 1], dvec[:, 0])
    dsafe = np.maximum(d, 1e-12)

    # Непрерывные поля ключа: (значение, ширина ячейки, период в ячейках или 0)
    fields: list[tuple[np.ndarray, float, np.ndarray | None]] = []
    # различные якоря
    lri = np.where(ki == KIND_TEXT, 0.0, np.log(np.maximum(si, 1e-12) / dsafe))
    lrj = np.where(kj == KIND_TEXT, 0.0, np.log(np.maximum(sj, 1e-12) / dsafe))
    per_i = _period(ki)
    per_j = _period(kj)
    ai = np.where(np.isnan(di), 0.0, np.mod(np.nan_to_num(di) - phi, per_i))
    aj = np.where(np.isnan(dj), 0.0, np.mod(np.nan_to_num(dj) - phi, per_j))
    # концентричные якоря
    lr_c = np.where(
        (ki == KIND_TEXT) | (kj == KIND_TEXT),
        0.0,
        np.log(np.maximum(si, 1e-12) / np.maximum(sj, 1e-12)),
    )
    per_c = np.where((ki == KIND_SEGMENT) | (kj == KIND_SEGMENT), math.pi, _TWO_PI)
    has_both = ~np.isnan(di) & ~np.isnan(dj)
    rel_c = np.where(has_both, np.mod(np.nan_to_num(dj) - np.nan_to_num(di), per_c), 0.0)

    f1 = np.where(conc, lr_c, lri)
    f2 = np.where(conc, 0.0, lrj)
    f3 = np.where(conc, rel_c, ai)
    f4 = np.where(conc, 0.0, aj)
    nb_ang_i = np.where(conc, np.round(per_c / ang_w), np.round(per_i / ang_w)).astype(np.int64)
    nb_ang_j = np.round(per_j / ang_w).astype(np.int64)
    fields = [
        (f1, rat_w, None),
        (f2, rat_w, None),
        (f3, ang_w, nb_ang_i),
        (f4, ang_w, nb_ang_j),
        (anchors.extent[i], ang_w, None),
        (anchors.extent[j], ang_w, None),
    ]

    base = np.zeros(len(i), dtype=np.uint64)
    base = _mix(base, conc.astype(np.int64))
    base = _mix(base, ki)
    base = _mix(base, kj)
    base = _mix(base, anchors.text_hash[i])
    base = _mix(base, anchors.text_hash[j])

    keys = base[None, :]
    for value, width, nb in fields:
        b0, b1 = _bins(value, width, spread)
        if nb is not None:
            b0 = _wrap_bins(b0, nb)
            b1 = _wrap_bins(b1, nb)
        k0 = _mix(keys, b0[None, :])
        if spread:
            k1 = _mix(keys, b1[None, :])
            # если соседняя ячейка совпала с основной (нет разброса), не дублировать
            keys = np.concatenate([k0, np.where((b1 == b0)[None, :], k0, k1)], axis=0)
        else:
            keys = k0
    rows = np.broadcast_to(np.arange(len(i))[None, :], keys.shape)
    keys_f = keys[:, valid].ravel().view(np.int64)
    rows_f = rows[:, valid].ravel()
    if spread:
        # убрать дубли (key, row)
        kr = np.stack([keys_f, rows_f], axis=1)
        kr = np.unique(kr, axis=0)
        return kr[:, 0].copy(), kr[:, 1].copy()
    return keys_f.copy(), rows_f.copy()


# Допуск склейки угловых интервалов кусков дуги одного штриха (радианы).
_ANGLE_EPS = 1e-6


def _merge_arc_intervals(pieces: list[Arc]) -> tuple[float, float] | None:
    """Угловой диапазон штриха по его кускам: (t0, t1) или None для полной окружности."""
    iv: list[tuple[float, float]] = []
    for p in pieces:
        if p.is_full:
            return None
        a = p.t0 % _TWO_PI
        b = a + (p.t1 - p.t0)
        if b > _TWO_PI:
            iv.append((a, _TWO_PI))
            iv.append((0.0, b - _TWO_PI))
        else:
            iv.append((a, b))
    iv.sort()
    merged: list[list[float]] = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + _ANGLE_EPS:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    covered = sum(b - a for a, b in merged)
    if covered >= _TWO_PI - 1e-3:
        return None
    # дуга = дополнение к самому большому промежутку
    gaps = []
    for k in range(len(merged)):
        end = merged[k][1]
        nxt = merged[(k + 1) % len(merged)][0] + (_TWO_PI if k == len(merged) - 1 else 0.0)
        gaps.append((nxt - end, k))
    _, k = max(gaps)
    t0 = merged[(k + 1) % len(merged)][0]
    t1 = merged[k][1]
    if t1 <= t0:
        t1 += _TWO_PI
    return t0, t1


def strokes_from_pieces(prims: Sequence[Primitive]) -> list[Primitive]:
    """Восстановить штрихи (примитивы до планаризации) из кусков по ``stroke_id``.

    Ключи пар считаются по штрихам: они не зависят от того, как планаризация разрезала
    линии (стена через символ режет его отрезки, касание диагонали режет окружность).
    Отрезки штриха -> один отрезок по крайним проекциям, дуги -> одна дуга или окружность,
    текст и смешанные группы остаются как есть.
    """
    groups: dict[int, list[Primitive]] = {}
    order: list[int] = []
    out_text: list[tuple[int, Primitive]] = []
    for pos, p in enumerate(prims):
        if isinstance(p, TextPrim):
            out_text.append((pos, p))
            continue
        sid = p.stroke_id if p.stroke_id >= 0 else -(p.id + 1)
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(p)
    out: list[Primitive] = []
    for sid in order:
        g = groups[sid]
        if len(g) == 1:
            out.append(g[0])
        elif all(isinstance(p, Segment) for p in g):
            longest = max(g, key=lambda p: math.hypot(p.p1[0] - p.p0[0], p.p1[1] - p.p0[1]))
            o = np.array(longest.p0)
            v = np.array(longest.p1) - o
            v = v / max(np.hypot(v[0], v[1]), 1e-12)
            pts = np.array([q for p in g for q in (p.p0, p.p1)], dtype=float)
            t = (pts - o) @ v
            a, b = o + t.min() * v, o + t.max() * v
            out.append(
                Segment(
                    longest.id,
                    (float(a[0]), float(a[1])),
                    (float(b[0]), float(b[1])),
                    longest.parent_id,
                    longest.style,
                    longest.stroke_id,
                )
            )
        elif all(isinstance(p, Arc) for p in g):
            longest = max(g, key=lambda p: p.r * (p.t1 - p.t0))
            rng = _merge_arc_intervals(g)  # type: ignore[arg-type]
            t0, t1 = (0.0, _TWO_PI) if rng is None else rng
            out.append(
                Arc(
                    longest.id,
                    longest.center,
                    longest.r,
                    t0,
                    t1,
                    longest.parent_id,
                    longest.style,
                    longest.stroke_id,
                )
            )
        else:
            out.extend(g)
    out.extend(p for _, p in out_text)
    return out
