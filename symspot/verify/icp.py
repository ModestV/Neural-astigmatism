"""3.4а ICP-уточнение гипотезы. Ядро (Opus)."""

from __future__ import annotations

import math

import numpy as np

from symspot.core.config import Config
from symspot.core.geometry_utils import umeyama
from symspot.core.types import Query, Similarity
from symspot.normalize.index import PrimIndex

# Число ближайших сэмплов чертежа, среди которых ищется сэмпл с согласованной касательной.
K_NEIGHBORS = 6
# Начальный радиус сопоставления ICP в долях диагонали экземпляра; сужается до cover_tol.
ICP_START_RADIUS = 0.15
# Для гипотез от текста: центр рамки текста смещен относительно глифов (асимметрия
# надстрочных и подстрочных), при повороте на 180° смещение удваивается.
ICP_START_RADIUS_TEXT = 0.35
MIN_MATCHES = 3
# Доля итераций ICP в режиме точка-прямая; остальные — точка-точка.
POINT_TO_LINE_FRAC = 0.66
# Отношение сингулярных чисел облака сэмплов, ниже которого эталон считается «линейным».
DEGENERATE_RATIO = 0.05


def match_points(
    pts: np.ndarray,
    tans: np.ndarray,
    samples: np.ndarray,
    sample_tans: np.ndarray,
    kdtree,
    radius: float | np.ndarray,
    cos_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой точки — ближайший сэмпл ближе ``radius`` с |cos| касательных >= cos_min.

    Возвращает (dist (N,), idx (N,)); при отсутствии dist = inf, idx = -1.
    """
    n = len(pts)
    dist_out = np.full(n, np.inf)
    idx_out = np.full(n, -1, dtype=np.int64)
    if n == 0 or kdtree is None or len(samples) == 0:
        return dist_out, idx_out
    k = min(K_NEIGHBORS, len(samples))
    rmax = float(np.max(radius))
    d, ii = kdtree.query(pts, k=k, distance_upper_bound=rmax)
    if k == 1:
        d = d[:, None]
        ii = ii[:, None]
    valid = np.isfinite(d) & (ii < len(samples))
    ii_safe = np.where(valid, ii, 0)
    cosv = np.abs((sample_tans[ii_safe] * tans[:, None, :]).sum(-1))
    r = np.broadcast_to(np.asarray(radius, dtype=float), (n,))[:, None]
    ok = valid & (cosv >= cos_min) & (d <= r)
    # первый подходящий по возрастанию расстояния
    has = ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    rows = np.flatnonzero(has)
    dist_out[rows] = d[rows, first[rows]]
    idx_out[rows] = ii[rows, first[rows]]
    return dist_out, idx_out


def is_degenerate(pts: np.ndarray) -> bool:
    """Точки почти на одной прямой: подобие по ним не определяется (масштаб, поворот)."""
    if len(pts) < 3:
        return True
    sv = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
    return bool(sv[1] < DEGENERATE_RATIO * sv[0])


def icp_refine(
    query: Query,
    T: Similarity,
    index: PrimIndex,
    cfg: Config,
    start_radius: float = ICP_START_RADIUS,
) -> tuple[Similarity, float]:
    """Уточнить T: сэмплы эталона -> ближайшие сэмплы чертежа (согласованные касательные,
    < ``icp_angle_deg``), Umeyama с сохранением ``T.mirror``, до ``icp_iters`` итераций.

    Возвращает (T, residual), residual — средний остаток по сопоставленным сэмплам в долях
    диагонали эталона (inf, если сопоставлений меньше 3).
    """
    if len(query.samples) < MIN_MATCHES or index.kdtree is None:
        return T, math.inf
    base = Similarity(1.0, 0.0, 0.0, 0.0, T.mirror)
    src = base.apply(query.samples)
    src_t = base.apply_vec(query.tangents)
    w = query.weights
    cur = Similarity(T.s, T.theta, T.tx, T.ty, False)
    degenerate = is_degenerate(query.samples)
    cos_min = math.cos(math.radians(cfg.icp_angle_deg))
    residual = math.inf
    n_it = max(1, cfg.icp_iters)
    for it in range(n_it):
        frac = it / max(1, n_it - 1)
        rad_rel = start_radius * (1 - frac) + 1.5 * cfg.cover_tol * frac
        radius = rad_rel * cur.s
        P = cur.apply(src)
        Pt = cur.apply_vec(src_t)
        d, idx = match_points(P, Pt, index.samples, index.tangents, index.kdtree, radius, cos_min)
        m = idx >= 0
        if m.sum() < MIN_MATCHES:
            return T, math.inf
        # точка на касательной прямой сэмпла чертежа, ближайшая к P (точка-прямая)
        q = index.samples[idx[m]]
        qt = index.tangents[idx[m]]
        if frac < POINT_TO_LINE_FRAC:
            proj = ((P[m] - q) * qt).sum(1)
            target = q + proj[:, None] * qt
        else:
            # точка-точка: фиксирует сдвиг вдоль линий (короткие отрезки, концы)
            target = q
        if degenerate:
            # геометрия на одной прямой: масштаб и поворот берутся из гипотезы
            shift = np.average(target - P[m], axis=0, weights=w[m])
            new = Similarity(cur.s, cur.theta, cur.tx + shift[0], cur.ty + shift[1], False)
        else:
            try:
                new = umeyama(src[m], target, allow_mirror=False, weights=w[m])
            except ValueError:
                return T, math.inf
        if new.s <= 0 or not math.isfinite(new.s):
            return T, math.inf
        delta = (
            abs(math.log(new.s / cur.s))
            + abs(math.atan2(math.sin(new.theta - cur.theta), math.cos(new.theta - cur.theta)))
            + math.hypot(new.tx - cur.tx, new.ty - cur.ty) / cur.s
        )
        cur = new
        residual = float(np.average(np.hypot(*(cur.apply(src[m]) - target).T), weights=w[m]))
        residual /= cur.s
        if delta < 1e-6 and frac >= 1.0:
            break
    return cur.compose(base), residual


def align_point_sets(
    src: np.ndarray,
    src_t: np.ndarray,
    dst: np.ndarray,
    dst_t: np.ndarray,
    T0: Similarity,
    cfg: Config,
    radius0: float,
    radius1: float,
) -> tuple[Similarity, float]:
    """ICP набора точек на набор (без индекса чертежа), с сохранением ``T0.mirror``.

    Радиус сопоставления сужается от ``radius0`` до ``radius1`` (в единицах ``dst``).
    Возвращает (T, chamfer), chamfer — средний остаток src -> dst после выравнивания
    по всем точкам src (несопоставленные считаются с ``radius0``).
    """
    from scipy.spatial import cKDTree

    if len(src) < MIN_MATCHES or len(dst) < MIN_MATCHES:
        return T0, math.inf
    tree = cKDTree(dst)
    base = Similarity(1.0, 0.0, 0.0, 0.0, T0.mirror)
    s_pts = base.apply(src)
    s_tan = base.apply_vec(src_t)
    cur = Similarity(T0.s, T0.theta, T0.tx, T0.ty, False)
    cos_min = math.cos(math.radians(cfg.icp_angle_deg))
    n_it = max(1, cfg.icp_iters)
    for it in range(n_it):
        frac = it / max(1, n_it - 1)
        radius = radius0 * (1 - frac) + radius1 * frac
        P = cur.apply(s_pts)
        d, idx = match_points(P, cur.apply_vec(s_tan), dst, dst_t, tree, radius, cos_min)
        m = idx >= 0
        if m.sum() < MIN_MATCHES:
            break
        try:
            new = umeyama(s_pts[m], dst[idx[m]], allow_mirror=False)
        except ValueError:
            break
        if not (math.isfinite(new.s) and new.s > 1e-9):
            break
        cur = new
    T = cur.compose(base)
    P = T.apply(src)
    d, idx = match_points(P, T.apply_vec(src_t), dst, dst_t, tree, radius0, cos_min)
    d = np.where(idx >= 0, d, radius0)
    return T, float(d.mean())
