"""Поиск точных копий эталона (без поворота и масштаба) по каноническому хешу. Задача T6."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from symspot.core.config import Config
from symspot.core.geometry_utils import (
    TWO_PI,
    normalize_text,
    prim_length,
    prims_bbox,
    transform_prim,
)
from symspot.core.types import (
    Arc,
    BBox,
    Hypothesis,
    Primitive,
    Query,
    Segment,
    Similarity,
    TextPrim,
)
from symspot.normalize.index import PrimIndex

_SEG, _ARC, _TXT = 0, 1, 2

# Допуск на длину отрезка / дугу при поиске кандидатов якоря: оба конца могут сдвинуться на tol.
_ENDS_FACTOR = 2.0

# Запасной якорь берется, если число его кандидатов <= RATIO * (число у основного) + SLACK.
_SECOND_ANCHOR_RATIO = 10
_SECOND_ANCHOR_SLACK = 10


class _Arrays:
    """Плоские массивы по примитивам чертежа (индексы совпадают с ``index.prims``)."""

    def __init__(self, index: PrimIndex) -> None:
        n = len(index.prims)
        self.kind = np.full(n, -1, dtype=np.int8)
        self.p0 = np.zeros((n, 2))
        self.p1 = np.zeros((n, 2))
        self.r = np.zeros(n)
        self.t0 = np.zeros(n)
        self.t1 = np.zeros(n)
        self.full = np.zeros(n, dtype=bool)
        self.text_center = np.zeros((n, 2))
        self.texts: dict[str, list[int]] = defaultdict(list)
        self.text_norm: list[str] = [""] * n
        for i, p in enumerate(index.prims):
            if isinstance(p, Segment):
                self.kind[i] = _SEG
                self.p0[i] = p.p0
                self.p1[i] = p.p1
            elif isinstance(p, Arc):
                self.kind[i] = _ARC
                self.p0[i] = p.center
                self.r[i] = p.r
                self.t0[i] = p.t0
                self.t1[i] = p.t1
                self.full[i] = p.is_full
            elif isinstance(p, TextPrim):
                self.kind[i] = _TXT
                b = p.bbox
                self.text_center[i] = (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))
                self.text_norm[i] = normalize_text(p.text)
                self.texts[self.text_norm[i]].append(i)
        self.seg_idx = np.flatnonzero(self.kind == _SEG)
        self.seg_len = np.hypot(
            self.p1[self.seg_idx, 0] - self.p0[self.seg_idx, 0],
            self.p1[self.seg_idx, 1] - self.p0[self.seg_idx, 1],
        )
        order = np.argsort(self.seg_len, kind="stable")
        self.seg_idx = self.seg_idx[order]
        self.seg_len = self.seg_len[order]
        self.arc_idx = np.flatnonzero(self.kind == _ARC)
        order = np.argsort(self.r[self.arc_idx], kind="stable")
        self.arc_idx = self.arc_idx[order]
        self.arc_r = self.r[self.arc_idx]


def _covered(lo: np.ndarray, hi: np.ndarray, length: float, tol: float) -> bool:
    """Объединение интервалов [lo, hi] покрывает [0, length] с зазорами и краями <= tol."""
    if len(lo) == 0:
        return False
    order = np.argsort(lo, kind="stable")
    lo = lo[order]
    hi = hi[order]
    reach = np.maximum.accumulate(hi)
    if lo[0] > tol:
        return False
    if len(lo) > 1 and np.any(lo[1:] > reach[:-1] + tol):
        return False
    return bool(reach[-1] >= length - tol)


def _segment_covered(
    q0: np.ndarray, q1: np.ndarray, win: np.ndarray, arr: _Arrays, tol: float
) -> bool:
    d = q1 - q0
    length = float(math.hypot(d[0], d[1]))
    u = d / length
    nrm = np.array([-u[1], u[0]])
    cand = win[arr.kind[win] == _SEG]
    if len(cand) == 0:
        return False
    a = arr.p0[cand] - q0
    b = arr.p1[cand] - q0
    ok = (np.abs(a @ nrm) <= tol) & (np.abs(b @ nrm) <= tol)
    if not ok.any():
        return False
    sa = (a @ u)[ok]
    sb = (b @ u)[ok]
    lo = np.minimum(sa, sb)
    hi = np.maximum(sa, sb)
    keep = (hi >= -tol) & (lo <= length + tol)
    return _covered(lo[keep], hi[keep], length, tol)


def _arc_covered(q: Arc, shift: np.ndarray, win: np.ndarray, arr: _Arrays, tol: float) -> bool:
    c = np.array(q.center) + shift
    cand = win[arr.kind[win] == _ARC]
    if len(cand) == 0:
        return False
    dc = np.hypot(arr.p0[cand, 0] - c[0], arr.p0[cand, 1] - c[1])
    ok = (dc <= tol) & (np.abs(arr.r[cand] - q.r) <= tol)
    cand = cand[ok]
    if len(cand) == 0:
        return False
    sweep = q.t1 - q.t0
    d = np.mod(arr.t0[cand] - q.t0, TWO_PI)
    sw = np.where(arr.full[cand], TWO_PI, arr.t1[cand] - arr.t0[cand])
    d = np.where(arr.full[cand], 0.0, d)
    lo = np.concatenate([d, d - TWO_PI])
    hi = np.concatenate([d + sw, d - TWO_PI + sw])
    lo_c = np.maximum(lo, 0.0)
    hi_c = np.minimum(hi, sweep)
    keep = hi_c >= lo_c - 1e-12
    ta = tol / max(q.r, tol)
    return _covered(lo_c[keep], hi_c[keep], sweep, ta)


def _text_matched(
    q: TextPrim, shift: np.ndarray, win: np.ndarray, arr: _Arrays, tol: float
) -> bool:
    cand = win[arr.kind[win] == _TXT]
    if len(cand) == 0:
        return False
    b = q.bbox
    c = np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])]) + shift
    s = normalize_text(q.text)
    for i in cand:
        if arr.text_norm[int(i)] == s and np.hypot(*(arr.text_center[i] - c)) <= tol:
            return True
    return False


def _anchor_rank(q: Primitive, arr: _Arrays, tol: float) -> tuple | None:
    """Ключ редкости примитива эталона: (число похожих на чертеже, приоритет, -размер)."""
    if isinstance(q, Segment):
        length = prim_length(q)
        if length <= 0:
            return None
        lo = np.searchsorted(arr.seg_len, length - _ENDS_FACTOR * tol, side="left")
        hi = np.searchsorted(arr.seg_len, length + _ENDS_FACTOR * tol, side="right")
        return (int(hi - lo), 0, -length)
    if isinstance(q, Arc):
        if q.r <= 0:
            return None
        lo = np.searchsorted(arr.arc_r, q.r - tol, side="left")
        hi = np.searchsorted(arr.arc_r, q.r + tol, side="right")
        return (int(hi - lo), 0, -prim_length(q))
    s = normalize_text(q.text)
    if not s:
        return None
    return (len(arr.texts.get(s, ())), 1, 0.0)


def _anchor_shifts(q: Primitive, arr: _Arrays, tol: float) -> np.ndarray:
    """Сдвиги (K, 2) = (якорь чертежа) - (якорь эталона) для кандидатов того же типа/размера."""
    if isinstance(q, Segment):
        a = np.array(q.p0)
        b = np.array(q.p1)
        d = b - a
        length = float(math.hypot(d[0], d[1]))
        u = d / length
        lo = np.searchsorted(arr.seg_len, length - _ENDS_FACTOR * tol, side="left")
        hi = np.searchsorted(arr.seg_len, length + _ENDS_FACTOR * tol, side="right")
        idx = arr.seg_idx[lo:hi]
        if len(idx) == 0:
            return np.empty((0, 2))
        ca = arr.p0[idx]
        cb = arr.p1[idx]
        v = cb - ca
        cross = np.abs(u[0] * v[:, 1] - u[1] * v[:, 0]) / np.maximum(arr.seg_len[lo:hi], 1e-12)
        ok = cross * length <= _ENDS_FACTOR * tol
        mid = 0.5 * (ca + cb)
        return mid[ok] - 0.5 * (a + b)
    if isinstance(q, Arc):
        lo = np.searchsorted(arr.arc_r, q.r - tol, side="left")
        hi = np.searchsorted(arr.arc_r, q.r + tol, side="right")
        idx = arr.arc_idx[lo:hi]
        if len(idx) == 0:
            return np.empty((0, 2))
        sweep = q.t1 - q.t0
        if q.is_full:
            ok = arr.full[idx]
        else:
            sw = arr.t1[idx] - arr.t0[idx]
            dt = np.mod(arr.t0[idx] - q.t0 + math.pi, TWO_PI) - math.pi
            ok = (~arr.full[idx]) & (np.abs(sw - sweep) * q.r <= _ENDS_FACTOR * tol)
            ok &= np.abs(dt) * q.r <= _ENDS_FACTOR * tol
        return arr.p0[idx][ok] - np.array(q.center)
    assert isinstance(q, TextPrim)
    s = normalize_text(q.text)
    idx = np.array(arr.texts.get(s, ()), dtype=np.int64)
    if len(idx) == 0:
        return np.empty((0, 2))
    b = q.bbox
    return arr.text_center[idx] - np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])])


def _matches(
    qprims: list[Primitive], shift: np.ndarray, win: np.ndarray, arr: _Arrays, tol: float
) -> bool:
    for q in qprims:
        if isinstance(q, Segment):
            q0 = np.array(q.p0) + shift
            q1 = np.array(q.p1) + shift
            if not _segment_covered(q0, q1, win, arr, tol):
                return False
        elif isinstance(q, Arc):
            if not _arc_covered(q, shift, win, arr, tol):
                return False
        elif not _text_matched(q, shift, win, arr, tol):
            return False
    return True


def find_exact(query: Query, index: PrimIndex, cfg: Config) -> list[Hypothesis]:
    """Неповернутые немасштабированные копии эталона.

    - Канонический вид набора примитивов: координаты в единицах чертежа относительно
      якоря, округленные до ``exact_tol * query.diag``, отсортированные; отрезки без
      учета направления, дуги по (центр, r, t0, t1), текст по (центр, строка).
    - Якорь = самый редкий на чертеже тип/размер примитива эталона; для каждого такого
      примитива чертежа берется окно (рамка эталона, приложенная к якорю), и сравнивается
      хеш окна. Окно может содержать лишние куски (стена через символ): сравнение по
      вхождению всех примитивов эталона, а не по равенству множеств.
    - Каждая копия -> Hypothesis(T = Similarity(query.diag, 0, cx, cy), source="exact",
      votes = число примитивов эталона). Сам эталон тоже находится.

    Реализация: вместо буквального хеша округленных координат сравнение идет с допуском
    ``tol`` (это снимает потерю совпадений на границах ячеек), отрезки и дуги эталона
    считаются найденными, если покрыты объединением коллинеарных кусков чертежа в окне.
    Копия, у которой разрезаны оба якоря (основной и запасной), этим путем не находится
    (ее найдет голосование).
    """
    if not query.prims or query.diag <= 0 or not index.prims:
        return []
    tol = cfg.exact_tol * query.diag
    qprims = [transform_prim(p, query.frame) for p in query.prims]
    arr = _Arrays(index)

    ranked: list[tuple[tuple, Primitive]] = []
    for q in qprims:
        key = _anchor_rank(q, arr, tol)
        if key is not None:
            ranked.append((key, q))
    if not ranked:
        return []
    ranked.sort(key=lambda kq: kq[0])
    # Основной якорь и, если он не намного дороже, запасной: копия с разрезанным основным
    # якорем (например, диагональ с Т-стыком) находится через запасной.
    anchors = [ranked[0][1]]
    c1 = ranked[0][0][0]
    for key, q in ranked[1:]:
        if key[0] <= _SECOND_ANCHOR_RATIO * c1 + _SECOND_ANCHOR_SLACK:
            anchors.append(q)
            break

    box: BBox | None = prims_bbox(qprims)
    assert box is not None
    if query.box is not None:
        corners = query.frame.apply(
            np.array([[query.box[0], query.box[1]], [query.box[2], query.box[3]]])
        )
        qb: BBox = (
            float(corners[0, 0]),
            float(corners[0, 1]),
            float(corners[1, 0]),
            float(corners[1, 1]),
        )
        box = (min(box[0], qb[0]), min(box[1], qb[1]), max(box[2], qb[2]), max(box[3], qb[3]))

    cx, cy = query.frame.tx, query.frame.ty
    found: list[np.ndarray] = []
    grid: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    cell = max(tol, 1e-12)

    def _duplicate(t: np.ndarray) -> bool:
        gx, gy = int(math.floor(t[0] / cell)), int(math.floor(t[1] / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for f in grid.get((gx + dx, gy + dy), ()):
                    if math.hypot(f[0] - t[0], f[1] - t[1]) <= tol:
                        return True
        return False

    for anchor in anchors:
        ordered = [anchor, *[q for q in qprims if q is not anchor]]
        for t in _anchor_shifts(anchor, arr, tol):
            if _duplicate(t):
                continue
            window: BBox = (
                box[0] + t[0] - tol,
                box[1] + t[1] - tol,
                box[2] + t[0] + tol,
                box[3] + t[1] + tol,
            )
            win = index.query_bbox(window)
            if len(win) == 0 or not _matches(ordered, t, win, arr, tol):
                continue
            found.append(t)
            grid[(int(math.floor(t[0] / cell)), int(math.floor(t[1] / cell)))].append(t)

    votes = float(len(query.prims))
    return [
        Hypothesis(
            legend_id=query.legend_id,
            T=Similarity(query.diag, 0.0, cx + float(t[0]), cy + float(t[1])),
            votes=votes,
            source="exact",
            variant=query.variant,
        )
        for t in found
    ]
