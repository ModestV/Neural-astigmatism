"""Склейка коллинеарных отрезков, удаление дублей и вырожденных. Задача T5."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from symspot.core.geometry_utils import prim_bbox
from symspot.core.types import Arc, Segment

_TWO_PI = 2.0 * math.pi
_EPS_LEN = 1e-12


def _candidate_pairs(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Пары (i < j) примитивов с пересекающимися рамками. boxes: (n, 4)."""
    n = len(boxes)
    if n < 2:
        e = np.empty(0, dtype=np.int64)
        return e, e
    geoms = shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    tree = shapely.STRtree(geoms)
    qi, ti = tree.query(geoms, predicate="intersects")
    m = qi < ti
    qi, ti = qi[m], ti[m]
    order = np.lexsort((ti, qi))
    return qi[order].astype(np.int64), ti[order].astype(np.int64)


def _merge_segments(
    segs: list[Segment], angle_tol_deg: float, gap: float
) -> list[tuple[int, Segment]]:
    """Склейка отрезков (уже без коротких): (индекс главного отрезка группы, результат)."""
    n = len(segs)
    if n < 2:
        return list(enumerate(segs))
    a = np.array([s.p0 for s in segs], dtype=float)
    b = np.array([s.p1 for s in segs], dtype=float)
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    u = d / length[:, None]
    boxes = np.column_stack(
        [
            np.minimum(a[:, 0], b[:, 0]) - gap,
            np.minimum(a[:, 1], b[:, 1]) - gap,
            np.maximum(a[:, 0], b[:, 0]) + gap,
            np.maximum(a[:, 1], b[:, 1]) + gap,
        ]
    )
    pi, pj = _candidate_pairs(boxes)
    if len(pi) == 0:
        return list(enumerate(segs))

    # ref = более длинный из пары (при равенстве меньший индекс), oth = второй
    swap = length[pj] > length[pi]
    ref = np.where(swap, pj, pi)
    oth = np.where(swap, pi, pj)
    ur = u[ref]
    uo = u[oth]
    cos_a = np.abs((ur * uo).sum(axis=1))
    sin_a = np.abs(ur[:, 0] * uo[:, 1] - ur[:, 1] * uo[:, 0])
    ang = np.arctan2(sin_a, cos_a)
    ang_ok = ang < math.radians(angle_tol_deg)

    def rel(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = p - a[ref]
        along = (v * ur).sum(axis=1)
        perp = np.abs(v[:, 0] * ur[:, 1] - v[:, 1] * ur[:, 0])
        return along, perp

    s0, h0 = rel(a[oth])
    s1, h1 = rel(b[oth])
    lo = np.minimum(s0, s1)
    hi = np.maximum(s0, s1)
    line_ok = (h0 < gap) & (h1 < gap)
    span_ok = (lo <= length[ref] + gap) & (hi >= -gap)
    collinear = ang_ok & line_ok & span_ok

    # дубли независимо от угла: концы совпадают в пределах gap (в любом направлении)
    def close(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.hypot(*(p - q).T) < gap

    same = close(a[pi], a[pj]) & close(b[pi], b[pj])
    flip = close(a[pi], b[pj]) & close(b[pi], a[pj])
    ok = collinear | same | flip
    if not ok.any():
        return list(enumerate(segs))

    graph = coo_matrix((np.ones(int(ok.sum())), (pi[ok], pj[ok])), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)

    # главный отрезок группы: самый длинный, при равенстве меньший индекс
    order = np.lexsort((np.arange(n), -length, labels))
    first_of = np.full(n_comp, -1, dtype=np.int64)
    seen = np.zeros(n_comp, dtype=bool)
    for idx in order:
        lab = labels[idx]
        if not seen[lab]:
            seen[lab] = True
            first_of[lab] = idx

    counts = np.bincount(labels, minlength=n_comp)
    members: dict[int, list[int]] = {}
    for i in range(n):
        lab = int(labels[i])
        if counts[lab] > 1:
            members.setdefault(lab, []).append(i)

    out: list[tuple[int, Segment]] = []
    for i in range(n):
        lab = int(labels[i])
        if first_of[lab] != i:
            continue
        if counts[lab] == 1:
            out.append((i, segs[i]))
            continue
        idxs = np.array(members[lab])
        ends = np.concatenate([a[idxs], b[idxs]])
        proj = ((ends - a[i]) * u[i]).sum(axis=1)
        p0 = ends[int(np.argmin(proj))]
        p1 = ends[int(np.argmax(proj))]
        s = segs[i]
        merged = Segment(
            s.id,
            (float(p0[0]), float(p0[1])),
            (float(p1[0]), float(p1[1])),
            s.parent_id,
            s.style,
            s.id,
        )
        out.append((i, merged))
    return out


def _dedupe_arcs(arcs: list[Arc], gap: float) -> list[int]:
    """Дубли дуг (центр, r, t0, t1 в пределах gap, gap / r): индексы оставшихся (первая)."""
    n = len(arcs)
    if n < 2:
        return list(range(n))
    c = np.array([x.center for x in arcs], dtype=float)
    r = np.array([x.r for x in arcs], dtype=float)
    t0 = np.array([x.t0 for x in arcs], dtype=float)
    t1 = np.array([x.t1 for x in arcs], dtype=float)
    full = t1 - t0 >= _TWO_PI - 1e-9
    boxes = np.array([prim_bbox(x) for x in arcs], dtype=float)
    boxes[:, :2] -= gap
    boxes[:, 2:] += gap
    pi, pj = _candidate_pairs(boxes)
    if len(pi) == 0:
        return list(range(n))

    def wrapdiff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dd = (x - y + math.pi) % _TWO_PI - math.pi
        return np.abs(dd)

    near = (np.hypot(*(c[pi] - c[pj]).T) < gap) & (np.abs(r[pi] - r[pj]) < gap)
    ang_tol = gap / np.maximum(np.minimum(r[pi], r[pj]), _EPS_LEN)
    same_rng = (wrapdiff(t0[pi], t0[pj]) < ang_tol) & (wrapdiff(t1[pi], t1[pj]) < ang_tol)
    dup = near & ((full[pi] & full[pj]) | (~full[pi] & ~full[pj] & same_rng))
    removed = np.zeros(n, dtype=bool)
    for i, j in zip(pi[dup].tolist(), pj[dup].tolist(), strict=True):
        if not removed[i] and not removed[j]:
            removed[j] = True
    return [k for k in range(n) if not removed[k]]


def merge_collinear(
    prims: Sequence[Segment | Arc],
    angle_tol_deg: float,
    gap: float,
    min_len: float,
) -> list[Segment | Arc]:
    """Склеить коллинеарные соприкасающиеся/перекрывающиеся отрезки.

    - Два отрезка склеиваются, если угол между ними < ``angle_tol_deg``, расстояние от
      концов одного до прямой другого < ``gap`` и зазор между проекциями < ``gap``
      (перекрытие тоже склеивается). Транзитивно (цепочки из 3+ отрезков).
    - Склеенный отрезок получает id и parent_id отрезка с наибольшей длиной в группе,
      ``stroke_id`` = своему id.
    - Точные дубли (отрезки и дуги с совпадающей геометрией в пределах ``gap``) удаляются.
    - Отрезки короче ``min_len`` и дуги с длиной < ``min_len`` удаляются.
    - Дуги не склеиваются (кроме удаления дублей). Порядок выхода детерминирован.
    - Без O(n²): кандидаты через STRtree или сетку.
    """
    thr = max(min_len, _EPS_LEN)
    segs: list[Segment] = []
    seg_pos: list[int] = []
    arcs: list[Arc] = []
    arc_pos: list[int] = []
    for k, p in enumerate(prims):
        if isinstance(p, Segment):
            if math.hypot(p.p1[0] - p.p0[0], p.p1[1] - p.p0[1]) >= thr:
                segs.append(p)
                seg_pos.append(k)
        elif isinstance(p, Arc) and p.r * (p.t1 - p.t0) >= thr:
            arcs.append(p)
            arc_pos.append(k)
    out: list[tuple[int, Segment | Arc]] = [
        (seg_pos[i], s) for i, s in _merge_segments(segs, angle_tol_deg, gap)
    ]
    out.extend((arc_pos[i], arcs[i]) for i in _dedupe_arcs(arcs, gap))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]
