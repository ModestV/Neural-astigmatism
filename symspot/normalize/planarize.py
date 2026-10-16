"""Планаризация: разрез отрезков и дуг в точках пересечения. Задача T5."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import shapely

from symspot.core.geometry_utils import prim_bbox
from symspot.core.types import Arc, Segment

_TWO_PI = 2.0 * math.pi
_EPS = 1e-12
_PARAM_EPS = 1e-9  # допуск параметра внутри отрезка при аналитическом пересечении
_PARALLEL_SIN = 1e-9


class _Cuts:
    """Накопитель точек разреза: (индекс примитива, длина дуги вдоль примитива от начала)."""

    def __init__(self) -> None:
        self.idx: list[np.ndarray] = []
        self.s: list[np.ndarray] = []

    def add(self, idx: np.ndarray, s: np.ndarray) -> None:
        if len(idx):
            self.idx.append(np.asarray(idx, dtype=np.int64))
            self.s.append(np.asarray(s, dtype=float))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.idx:
            return np.empty(0, dtype=np.int64), np.empty(0)
        return np.concatenate(self.idx), np.concatenate(self.s)


def _arc_offset(
    T0: np.ndarray,
    T1: np.ndarray,
    R: np.ndarray,
    full: np.ndarray,
    arc_idx: np.ndarray,
    theta: np.ndarray,
    snap_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Принадлежность угла дуге: (маска, смещение угла от t0 в радианах, обрезанное)."""
    t0 = T0[arc_idx]
    span = T1[arc_idx] - t0
    dth = (theta - t0) % _TWO_PI
    tol = snap_tol / np.maximum(R[arc_idx], _EPS)
    wrap_before = dth >= _TWO_PI - tol
    ok = full[arc_idx] | (dth <= span + tol) | wrap_before
    off = np.where(wrap_before & ~full[arc_idx], 0.0, np.minimum(dth, span))
    off = np.where(full[arc_idx], dth, off)
    return ok, off


def planarize(
    prims: Sequence[Segment | Arc],
    snap_tol: float,
    min_len: float,
    start_id: int = 0,
) -> list[Segment | Arc]:
    """Разрезать все примитивы во всех точках взаимного пересечения и касания концом.

    - Кандидаты пар через ``shapely.STRtree`` по рамкам; пересечения аналитически:
      отрезок-отрезок, отрезок-дуга, дуга-дуга (включая полные окружности).
    - Конец одного примитива, лежащий на другом ближе ``snap_tol`` (Т-стык), режет другой.
    - Каждый кусок: новый id подряд с ``start_id``, ``parent_id`` и ``style`` исходного,
      ``stroke_id`` = id исходного примитива. Куски короче ``min_len`` отбрасываются.
    - Кусок полной окружности -> Arc с t0 < t1 (t1 - t0 < 2π), если окружность разрезана
      хотя бы в двух точках; разрезанная в одной точке окружность остается полной.
    - Сложность ~ O(n log n + k). Порядок выхода детерминирован.
    """
    n = len(prims)
    if n == 0:
        return []
    thr = max(min_len, _EPS)

    is_arc = np.array([isinstance(p, Arc) for p in prims], dtype=bool)
    A = np.zeros((n, 2))  # начало отрезка
    D = np.zeros((n, 2))  # вектор отрезка
    C = np.zeros((n, 2))  # центр дуги
    R = np.zeros(n)
    T0 = np.zeros(n)
    T1 = np.zeros(n)
    boxes = np.zeros((n, 4))
    for k, p in enumerate(prims):
        if isinstance(p, Segment):
            A[k] = p.p0
            D[k] = (p.p1[0] - p.p0[0], p.p1[1] - p.p0[1])
            boxes[k] = (
                min(p.p0[0], p.p1[0]),
                min(p.p0[1], p.p1[1]),
                max(p.p0[0], p.p1[0]),
                max(p.p0[1], p.p1[1]),
            )
        else:
            C[k] = p.center
            R[k] = p.r
            T0[k] = p.t0
            T1[k] = p.t1
            boxes[k] = prim_bbox(p)
    seg_len = np.hypot(D[:, 0], D[:, 1])
    full = is_arc & (T1 - T0 >= _TWO_PI - 1e-9)
    total = np.where(is_arc, R * (T1 - T0), seg_len)  # длина каждого примитива
    valid = total >= thr

    cuts = _Cuts()
    vidx = np.flatnonzero(valid)
    if len(vidx) >= 2:
        vb = boxes[vidx]
        geoms = shapely.box(
            vb[:, 0] - snap_tol, vb[:, 1] - snap_tol, vb[:, 2] + snap_tol, vb[:, 3] + snap_tol
        )
        tree = shapely.STRtree(geoms)
        qi, ti = tree.query(geoms, predicate="intersects")
        m = qi < ti
        pi = vidx[qi[m]]
        pj = vidx[ti[m]]
        order = np.lexsort((pj, pi))
        pi, pj = pi[order], pj[order]
        arc_i = is_arc[pi]
        arc_j = is_arc[pj]
        ss = ~arc_i & ~arc_j
        sa_a = ~arc_i & arc_j
        sa_b = arc_i & ~arc_j
        _seg_seg(pi[ss], pj[ss], A, D, seg_len, snap_tol, cuts)
        # отрезок-дуга: отрезок всегда первый
        sidx = np.concatenate([pi[sa_a], pj[sa_b]])
        aidx = np.concatenate([pj[sa_a], pi[sa_b]])
        _seg_arc(sidx, aidx, A, D, seg_len, C, R, T0, T1, full, snap_tol, cuts)
        aa = arc_i & arc_j
        _arc_arc(pi[aa], pj[aa], C, R, T0, T1, full, snap_tol, cuts)

    return _build_pieces(prims, is_arc, total, full, cuts, snap_tol, thr, start_id)


# ---------------------------------------------------------------------------
# Пересечения
# ---------------------------------------------------------------------------


def _point_on_seg(
    pts: np.ndarray, sidx: np.ndarray, A: np.ndarray, D: np.ndarray, L: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Точки pts (k, 2) против отрезков sidx: (маска расстояние < tol, длина до проекции)."""
    ll = L[sidx]
    v = pts - A[sidx]
    t = np.clip((v * D[sidx]).sum(axis=1) / (ll * ll), 0.0, 1.0)
    foot = A[sidx] + D[sidx] * t[:, None]
    dist = np.hypot(*(pts - foot).T)
    return dist < tol, t * ll


def _seg_seg(
    si: np.ndarray,
    sj: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    tol: float,
    cuts: _Cuts,
) -> None:
    if len(si) == 0:
        return
    r = D[si]
    s = D[sj]
    qp = A[sj] - A[si]
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    nonpar = np.abs(den) > _PARALLEL_SIN * L[si] * L[sj]
    safe = np.where(nonpar, den, 1.0)
    t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / safe
    u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / safe
    hit = nonpar & (t >= -_PARAM_EPS) & (t <= 1 + _PARAM_EPS) & (u >= -_PARAM_EPS)
    hit &= u <= 1 + _PARAM_EPS
    cuts.add(si[hit], np.clip(t[hit], 0.0, 1.0) * L[si[hit]])
    cuts.add(sj[hit], np.clip(u[hit], 0.0, 1.0) * L[sj[hit]])
    # концы одного на другом (Т-стык, касание, коллинеарное перекрытие)
    for src, dst in ((si, sj), (sj, si)):
        for pts in (A[src], A[src] + D[src]):
            ok, sd = _point_on_seg(pts, dst, A, D, L, tol)
            cuts.add(dst[ok], sd[ok])


def _seg_arc(
    si: np.ndarray,
    ai: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    T0: np.ndarray,
    T1: np.ndarray,
    full: np.ndarray,
    tol: float,
    cuts: _Cuts,
) -> None:
    if len(si) == 0:
        return
    a = A[si]
    d = D[si]
    ll = L[si]
    c = C[ai]
    rad = R[ai]
    # пересечение прямой отрезка с окружностью
    tstar = -((a - c) * d).sum(axis=1) / (ll * ll)
    foot = a + d * tstar[:, None]
    dist = np.hypot(*(foot - c).T)
    h2 = rad * rad - dist * dist
    near_tangent = (h2 < 0) & (dist - rad < tol)
    ok_root = (h2 >= 0) | near_tangent
    hs = np.sqrt(np.maximum(h2, 0.0)) / ll
    for sign in (-1.0, 1.0):
        t = tstar + sign * hs
        m = ok_root & (t >= -_PARAM_EPS) & (t <= 1 + _PARAM_EPS)
        if not m.any():
            continue
        tc = np.clip(t[m], 0.0, 1.0)
        pt = a[m] + d[m] * tc[:, None]
        theta = np.arctan2(pt[:, 1] - c[m][:, 1], pt[:, 0] - c[m][:, 0])
        inrng, off = _arc_offset(T0, T1, R, full, ai[m], theta, tol)
        cuts.add(si[m][inrng], tc[inrng] * ll[m][inrng])
        cuts.add(ai[m][inrng], off[inrng] * rad[m][inrng])
    # концы отрезка на дуге
    for pts in (a, a + d):
        rel = pts - c
        near = np.abs(np.hypot(rel[:, 0], rel[:, 1]) - rad) < tol
        if not near.any():
            continue
        theta = np.arctan2(rel[near][:, 1], rel[near][:, 0])
        inrng, off = _arc_offset(T0, T1, R, full, ai[near], theta, tol)
        cuts.add(ai[near][inrng], off[inrng] * rad[near][inrng])
    # концы дуги на отрезке
    part = ~full[ai]
    for ang in (T0, T1):
        sel = np.flatnonzero(part)
        if len(sel) == 0:
            continue
        th = ang[ai[sel]]
        pts = c[sel] + rad[sel][:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
        ok, sd = _point_on_seg(pts, si[sel], A, D, L, tol)
        cuts.add(si[sel][ok], sd[ok])


def _arc_arc(
    ai: np.ndarray,
    aj: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    T0: np.ndarray,
    T1: np.ndarray,
    full: np.ndarray,
    tol: float,
    cuts: _Cuts,
) -> None:
    if len(ai) == 0:
        return
    ci, cj = C[ai], C[aj]
    ri, rj = R[ai], R[aj]
    v = cj - ci
    d = np.hypot(v[:, 0], v[:, 1])
    ok = (d > _EPS) & (d <= ri + rj + tol) & (d >= np.abs(ri - rj) - tol)
    if ok.any():
        sel = np.flatnonzero(ok)
        dd = d[sel]
        e = v[sel] / dd[:, None]
        a_ = (ri[sel] ** 2 - rj[sel] ** 2 + dd * dd) / (2 * dd)
        hs = np.sqrt(np.maximum(ri[sel] ** 2 - a_**2, 0.0))
        base = ci[sel] + e * a_[:, None]
        perp = np.stack([-e[:, 1], e[:, 0]], axis=1)
        for sign in (-1.0, 1.0):
            pt = base + perp * (sign * hs)[:, None]
            th_i = np.arctan2(pt[:, 1] - ci[sel][:, 1], pt[:, 0] - ci[sel][:, 0])
            th_j = np.arctan2(pt[:, 1] - cj[sel][:, 1], pt[:, 0] - cj[sel][:, 0])
            ok_i, off_i = _arc_offset(T0, T1, R, full, ai[sel], th_i, tol)
            ok_j, off_j = _arc_offset(T0, T1, R, full, aj[sel], th_j, tol)
            both = ok_i & ok_j
            cuts.add(ai[sel][both], off_i[both] * ri[sel][both])
            cuts.add(aj[sel][both], off_j[both] * rj[sel][both])
    # концы одной дуги на другой
    for src, dst in ((ai, aj), (aj, ai)):
        part = np.flatnonzero(~full[src])
        if len(part) == 0:
            continue
        for ang in (T0, T1):
            th = ang[src[part]]
            cs = C[src[part]]
            pts = cs + R[src[part]][:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
            rel = pts - C[dst[part]]
            dist = np.hypot(rel[:, 0], rel[:, 1])
            near = np.abs(dist - R[dst[part]]) < tol
            if not near.any():
                continue
            theta = np.arctan2(rel[near][:, 1], rel[near][:, 0])
            didx = dst[part][near]
            inrng, off = _arc_offset(T0, T1, R, full, didx, theta, tol)
            cuts.add(didx[inrng], off[inrng] * R[didx][inrng])


# ---------------------------------------------------------------------------
# Сборка кусков
# ---------------------------------------------------------------------------


def _build_pieces(
    prims: Sequence[Segment | Arc],
    is_arc: np.ndarray,
    total: np.ndarray,
    full: np.ndarray,
    cuts: _Cuts,
    snap_tol: float,
    thr: float,
    start_id: int,
) -> list[Segment | Arc]:
    n = len(prims)
    cidx, cs = cuts.arrays()
    # отфильтровать разрезы у концов незамкнутых примитивов
    keep = np.ones(len(cidx), dtype=bool)
    if len(cidx):
        tot = total[cidx]
        open_ = ~full[cidx]
        keep = ~(open_ & ((cs < snap_tol) | (cs > tot - snap_tol)))
        keep &= tot >= thr
        cidx, cs = cidx[keep], cs[keep]
        order = np.lexsort((cs, cidx))
        cidx, cs = cidx[order], cs[order]
        # жадная прореженность: разрез остается, если от предыдущего исходного >= snap_tol
        if len(cidx):
            same = np.concatenate([[False], cidx[1:] == cidx[:-1]])
            gap_ok = np.concatenate([[True], (cs[1:] - cs[:-1]) >= snap_tol])
            keep = ~same | gap_ok
            cidx, cs = cidx[keep], cs[keep]
    starts = np.searchsorted(cidx, np.arange(n + 1)) if len(cidx) else np.zeros(n + 1, int)

    out: list[Segment | Arc] = []
    next_id = start_id

    def sub_seg(p: Segment, s0: float, s1: float, ll: float, whole: tuple[bool, bool]) -> None:
        nonlocal next_id
        if s1 - s0 < thr:
            return
        ax, ay = p.p0
        dx, dy = p.p1[0] - ax, p.p1[1] - ay
        q0 = p.p0 if whole[0] else (ax + dx * (s0 / ll), ay + dy * (s0 / ll))
        q1 = p.p1 if whole[1] else (ax + dx * (s1 / ll), ay + dy * (s1 / ll))
        out.append(Segment(next_id, q0, q1, p.parent_id, p.style, p.id))
        next_id += 1

    def sub_arc(p: Arc, a0: float, a1: float) -> None:
        nonlocal next_id
        if (a1 - a0) * p.r < thr:
            return
        t0, t1 = p.t0 + a0, p.t0 + a1
        if t0 >= _TWO_PI or t0 < 0.0:
            shift = math.floor(t0 / _TWO_PI) * _TWO_PI
            t0, t1 = t0 - shift, t1 - shift
        out.append(Arc(next_id, p.center, p.r, t0, t1, p.parent_id, p.style, p.id))
        next_id += 1

    for k in range(n):
        p = prims[k]
        if total[k] < thr:
            continue
        lo, hi = starts[k], starts[k + 1]
        ks = cs[lo:hi]
        if isinstance(p, Segment):
            ll = float(total[k])
            if len(ks) == 0:
                out.append(Segment(next_id, p.p0, p.p1, p.parent_id, p.style, p.id))
                next_id += 1
                continue
            bounds = [0.0, *ks.tolist(), ll]
            nb = len(bounds)
            for m in range(nb - 1):
                sub_seg(p, bounds[m], bounds[m + 1], ll, (m == 0, m == nb - 2))
            continue
        # дуга
        r = p.r
        if full[k]:
            offs = ks.tolist()
            if offs and (_TWO_PI * r - offs[-1]) + offs[0] < snap_tol:
                offs.pop()
            if len(offs) <= 1:
                out.append(Arc(next_id, p.center, r, p.t0, p.t1, p.parent_id, p.style, p.id))
                next_id += 1
                continue
            ang = [o / r for o in offs]
            for m in range(len(ang)):
                a0 = ang[m]
                a1 = ang[m + 1] if m + 1 < len(ang) else ang[0] + _TWO_PI
                sub_arc(p, a0, a1)
            continue
        if len(ks) == 0:
            out.append(Arc(next_id, p.center, r, p.t0, p.t1, p.parent_id, p.style, p.id))
            next_id += 1
            continue
        span = p.t1 - p.t0
        bounds = [0.0, *(ks / r).tolist(), span]
        for m in range(len(bounds) - 1):
            sub_arc(p, bounds[m], bounds[m + 1])
    return out
