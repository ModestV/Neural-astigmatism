import math

import numpy as np
import pytest

from symspot.core.config import Config
from symspot.core.types import Arc, Primitive, QuerySpec, Segment, TextPrim
from symspot.exact.hashing import find_exact
from symspot.normalize.index import PrimIndex
from symspot.query.extract import extract_query

R = 10.0
D = R / math.sqrt(2.0)
CFG = Config()
HALF = R + 1.0
_next_id = [0]


def _id() -> int:
    _next_id[0] += 1
    return _next_id[0]


def _circle(cx: float, cy: float, r: float = R) -> Arc:
    i = _id()
    return Arc(i, (cx, cy), r, 0.0, 2 * math.pi, i)


def _seg(p0, p1) -> Segment:
    i = _id()
    return Segment(i, tuple(p0), tuple(p1), i)


def _slash(cx: float, cy: float, d: float = D) -> Segment:
    return _seg((cx - d, cy + d), (cx + d, cy - d))


def _backslash(cx: float, cy: float, d: float = D) -> Segment:
    return _seg((cx - d, cy - d), (cx + d, cy + d))


def _sym(cx: float, cy: float) -> list[Primitive]:
    """Символ: круг с диагональю."""
    return [_circle(cx, cy), _slash(cx, cy)]


def _text(s: str, cx: float, cy: float) -> TextPrim:
    i = _id()
    return TextPrim(i, s, (cx - 2, cy - 1, cx + 2, cy + 1), i)


def _build(prims: list[Primitive], ref_center: tuple[float, float]):
    index = PrimIndex(prims, 1.0)
    cx, cy = ref_center
    spec = QuerySpec("L", (cx - HALF, cy - HALF, cx + HALF, cy + HALF))
    query = extract_query(spec, index, CFG)
    return query, index


def _centers(hyps):
    return sorted((h.T.tx, h.T.ty) for h in hyps)


def _assert_centers(hyps, expected):
    got = _centers(hyps)
    exp = sorted(expected)
    assert len(got) == len(exp), got
    tol = CFG.exact_tol * hyps[0].T.s if hyps else 0.0
    for g, e in zip(got, exp, strict=True):
        assert math.hypot(g[0] - e[0], g[1] - e[1]) <= tol + 1e-9


def _rot90(prims: list[Primitive], c: tuple[float, float]) -> list[Primitive]:
    out: list[Primitive] = []
    for p in prims:
        if isinstance(p, Segment):
            f = [(c[0] - (y - c[1]), c[1] + (x - c[0])) for x, y in (p.p0, p.p1)]
            out.append(_seg(f[0], f[1]))
        else:
            assert isinstance(p, Arc)
            out.append(_circle(*p.center, p.r))
    return out


def test_three_exact_copies_plus_reference():
    centers = [(50.0, 50.0), (150.0, 40.0), (80.0, 160.0), (200.0, 200.0)]
    prims = [p for c in centers for p in _sym(*c)]
    q, idx = _build(prims, centers[0])
    hyps = find_exact(q, idx, CFG)
    assert len(hyps) == 4
    _assert_centers(hyps, centers)
    h = hyps[0]
    assert h.source == "exact" and h.legend_id == "L" and h.T.s == pytest.approx(q.diag)
    assert h.T.theta == 0.0 and not h.T.mirror
    assert h.votes == float(len(q.prims))


def test_reference_is_found_with_zero_shift():
    prims = _sym(50.0, 50.0)
    q, idx = _build(prims, (50.0, 50.0))
    hyps = find_exact(q, idx, CFG)
    assert len(hyps) == 1
    assert hyps[0].T.tx == pytest.approx(q.frame.tx, abs=1e-9)
    assert hyps[0].T.ty == pytest.approx(q.frame.ty, abs=1e-9)


def test_rotated_and_scaled_copies_not_found():
    ref = _sym(50.0, 50.0)
    rot = _rot90(_sym(150.0, 50.0), (150.0, 50.0))
    scaled = [_circle(250.0, 50.0, R * 1.1), _slash(250.0, 50.0, D * 1.1)]
    q, idx = _build(ref + rot + scaled, (50.0, 50.0))
    hyps = find_exact(q, idx, CFG)
    _assert_centers(hyps, [(50.0, 50.0)])


def test_extra_line_through_copy_is_ok():
    wall = _seg((100.0, 100.0), (230.0, 100.0))
    prims = _sym(50.0, 50.0) + _sym(150.0, 100.0) + [wall]
    q, idx = _build(prims, (50.0, 50.0))
    hyps = find_exact(q, idx, CFG)
    _assert_centers(hyps, [(50.0, 50.0), (150.0, 100.0)])


def test_copy_missing_a_line_not_found():
    no_diag = [_circle(150.0, 50.0)]
    no_circle = [_slash(250.0, 50.0)]
    prims = _sym(50.0, 50.0) + no_diag + no_circle
    q, idx = _build(prims, (50.0, 50.0))
    _assert_centers(find_exact(q, idx, CFG), [(50.0, 50.0)])


def test_text_distinguishes_symbols():
    prims = [
        _circle(50.0, 50.0),
        _text("SD", 50.0, 50.0),
        _circle(150.0, 50.0),
        _text("sd", 150.0, 50.0),  # регистр не важен
        _circle(250.0, 50.0),
        _text("CO", 250.0, 50.0),
        _circle(350.0, 50.0),  # без текста
    ]
    q, idx = _build(prims, (50.0, 50.0))
    assert len(q.texts) == 1
    hyps = find_exact(q, idx, CFG)
    _assert_centers(hyps, [(50.0, 50.0), (150.0, 50.0)])


def test_text_shifted_not_found():
    prims = [_circle(50.0, 50.0), _text("SD", 50.0, 50.0), _circle(150.0, 50.0)]
    prims.append(_text("SD", 150.0 + 3.0, 50.0))
    q, idx = _build(prims, (50.0, 50.0))
    _assert_centers(find_exact(q, idx, CFG), [(50.0, 50.0)])


def test_subset_relation_slash_in_x():
    slash_circle = _sym(50.0, 50.0)
    x_sym = [_circle(150.0, 50.0), _slash(150.0, 50.0), _backslash(150.0, 50.0)]
    # эталон: круг с диагональю, находит и себя, и X (X содержит его целиком)
    q, idx = _build(slash_circle + x_sym, (50.0, 50.0))
    _assert_centers(find_exact(q, idx, CFG), [(50.0, 50.0), (150.0, 50.0)])
    # эталон: X, круг с диагональю X не содержит
    q2, idx2 = _build(slash_circle + x_sym, (150.0, 50.0))
    _assert_centers(find_exact(q2, idx2, CFG), [(150.0, 50.0)])


def test_split_collinear_pieces_found():
    ref = _sym(50.0, 50.0)
    # круг копии разрезан на 3 дуги, диагональ цела; одинаковые круги в других местах
    # делают якорем диагональ
    cx, cy = 150.0, 50.0
    pieces = [
        _slash(cx, cy),
        Arc(_id(), (cx, cy), R, 0.0, 2.0, 1),
        Arc(_id(), (cx, cy), R, 2.0, 5.0, 1),
        Arc(_id(), (cx, cy), R, 5.0, 2 * math.pi, 1),
        _seg((cx - 30, cy + 20.0), (cx + 30, cy + 20.0)),  # лишнее
    ]
    decoys = [_circle(300.0, 300.0), _circle(400.0, 300.0)]
    q, idx = _build(ref + pieces + decoys, (50.0, 50.0))
    _assert_centers(find_exact(q, idx, CFG), [(50.0, 50.0), (cx, cy)])


def test_both_anchors_split_not_found():
    ref = _sym(50.0, 50.0)
    cx, cy = 150.0, 50.0
    pieces = [
        _seg((cx - D, cy + D), (cx, cy)),
        _seg((cx, cy), (cx + D, cy - D)),
        Arc(_id(), (cx, cy), R, 0.0, 2.0, 1),
        Arc(_id(), (cx, cy), R, 2.0, 2 * math.pi, 1),
    ]
    q, idx = _build(ref + pieces, (50.0, 50.0))
    _assert_centers(find_exact(q, idx, CFG), [(50.0, 50.0)])


def test_split_diagonal_with_whole_circle_found():
    ref = _sym(50.0, 50.0)
    cx, cy = 150.0, 50.0
    mid = (cx - 2.0, cy + 2.0)
    pieces = [
        _circle(cx, cy),
        _seg((cx - D, cy + D), mid),
        _seg(mid, (cx + D, cy - D)),
    ]
    q, idx = _build(ref + pieces, (50.0, 50.0))
    _assert_centers(find_exact(q, idx, CFG), [(50.0, 50.0), (cx, cy)])


def test_gap_in_split_pieces_not_found():
    ref = _sym(50.0, 50.0)
    cx, cy = 150.0, 50.0
    gap = 1.0  # заметно больше tol (~0.16)
    pieces = [
        _circle(cx, cy),
        _seg((cx - D, cy + D), (cx - 0.5 * gap * 0.7, cy + 0.5 * gap * 0.7)),
        _seg((cx + 0.5 * gap * 0.7, cy - 0.5 * gap * 0.7), (cx + D, cy - D)),
    ]
    q, idx = _build(ref + pieces, (50.0, 50.0))
    _assert_centers(find_exact(q, idx, CFG), [(50.0, 50.0)])


def test_small_noise_found_large_noise_not():
    tol = CFG.exact_tol * 2 * math.hypot(HALF, HALF)
    ref = _sym(50.0, 50.0)
    near = [
        _circle(150.0 + 0.4 * tol, 50.0 - 0.3 * tol),
        _slash(150.0 + 0.4 * tol, 50.0 - 0.3 * tol),
    ]
    far = [_circle(250.0, 50.0), _slash(250.0 + 5 * tol, 50.0)]
    q, idx = _build(ref + near + far, (50.0, 50.0))
    hyps = find_exact(q, idx, CFG)
    _assert_centers(hyps, [(50.0, 50.0), (150.0 + 0.4 * tol, 50.0 - 0.3 * tol)])


def test_no_duplicate_hypotheses_and_variant():
    prims = _sym(50.0, 50.0) + _sym(150.0, 50.0)
    q, idx = _build(prims, (50.0, 50.0))
    q.variant = 3
    hyps = find_exact(q, idx, CFG)
    assert len(hyps) == 2
    assert all(h.variant == 3 for h in hyps)


def test_empty_query_and_empty_drawing():
    q, idx = _build(_sym(50.0, 50.0), (50.0, 50.0))
    assert find_exact(q, PrimIndex([], 1.0), CFG) == []
    q.prims = []
    assert find_exact(q, idx, CFG) == []


@pytest.mark.slow
def test_performance_30k_random_segments():
    import time

    rng = np.random.default_rng(0)
    n = 30000
    p0 = rng.uniform(0, 3000, size=(n, 2))
    ang = rng.uniform(0, 2 * math.pi, size=n)
    ln = rng.uniform(1.0, 30.0, size=n)
    p1 = p0 + np.stack([np.cos(ang), np.sin(ang)], 1) * ln[:, None]
    prims: list[Primitive] = [_seg(a, b) for a, b in zip(p0, p1, strict=True)]
    centers = [(50.0 + 14.0 * (i % 200), 3100.0 + 30.0 * (i // 200)) for i in range(200)]
    # 200 копий на отдельных позициях, эталон = первая копия
    prims += [p for c in centers for p in _sym(*c)]
    q, idx = _build(prims, centers[0])
    t = time.perf_counter()
    hyps = find_exact(q, idx, CFG)
    dt = time.perf_counter() - t
    assert len(hyps) == 200
    assert dt < 2.0, dt
