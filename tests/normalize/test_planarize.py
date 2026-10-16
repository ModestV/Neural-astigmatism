import math

import numpy as np
import pytest

from symspot.core.geometry_utils import arc_point
from symspot.core.types import DEFAULT_STYLE, Arc, Segment, Style
from symspot.normalize.planarize import planarize

TOL = 0.002
MIN = 1e-4
TWO_PI = 2 * math.pi


def S(i, p0, p1, parent=None, style=DEFAULT_STYLE):
    return Segment(i, p0, p1, i if parent is None else parent, style, i)


def _by_stroke(out, sid):
    return [p for p in out if p.stroke_id == sid]


def test_cross_gives_four_pieces():
    out = planarize([S(0, (-1, 0), (1, 0)), S(1, (0, -1), (0, 1))], TOL, MIN)
    assert len(out) == 4
    assert all(isinstance(p, Segment) for p in out)
    assert all(math.isclose(math.dist(p.p0, p.p1), 1.0) for p in out)
    assert sorted(p.id for p in out) == [0, 1, 2, 3]


def test_start_id_and_attributes():
    st = Style(stroke_width=1.5)
    out = planarize(
        [S(7, (-1, 0), (1, 0), parent=3, style=st), S(8, (0, -1), (0, 1))], TOL, MIN, 50
    )
    assert [p.id for p in out] == [50, 51, 52, 53]
    first = _by_stroke(out, 7)
    assert len(first) == 2 and all(p.parent_id == 3 and p.style == st for p in first)
    assert all(p.parent_id == 8 for p in _by_stroke(out, 8))


def test_t_junction_within_snap_tol():
    # вертикаль заканчивается на 0.001 выше горизонтали
    out = planarize([S(0, (-5, 0), (5, 0)), S(1, (0, 3), (0, 0.001))], TOL, MIN)
    horiz = _by_stroke(out, 0)
    assert len(horiz) == 2
    xs = sorted([p.p0[0] for p in horiz] + [p.p1[0] for p in horiz])
    assert xs == pytest.approx([-5, 0, 0, 5])
    assert len(_by_stroke(out, 1)) == 1


def test_t_junction_too_far_does_not_cut():
    out = planarize([S(0, (-5, 0), (5, 0)), S(1, (0, 3), (0, 0.5))], TOL, MIN)
    assert len(out) == 2


def test_l_corner_not_cut():
    out = planarize([S(0, (0, 0), (5, 0)), S(1, (0, 0), (0, 5))], TOL, MIN)
    assert len(out) == 2


def test_collinear_overlap_cut_at_endpoints():
    out = planarize([S(0, (0, 0), (6, 0)), S(1, (4, 0), (10, 0))], TOL, MIN)
    a = sorted((p.p0[0], p.p1[0]) for p in _by_stroke(out, 0))
    b = sorted((p.p0[0], p.p1[0]) for p in _by_stroke(out, 1))
    assert a == [(0, 4), (4, 6)]
    assert b == [(4, 6), (6, 10)]


def test_wall_through_symbol():
    circle = Arc(1, (0, 0), 1.0, 0.0, TWO_PI, 1, Style(), 1)
    wall = S(0, (-5, 0), (5, 0))
    out = planarize([wall, circle], TOL, MIN)
    walls = sorted(_by_stroke(out, 0), key=lambda p: p.p0[0])
    assert len(walls) == 3
    assert walls[1].p0 == pytest.approx((-1, 0)) and walls[1].p1 == pytest.approx((1, 0))
    arcs = _by_stroke(out, 1)
    assert len(arcs) == 2 and all(isinstance(a, Arc) for a in arcs)
    assert sum(a.t1 - a.t0 for a in arcs) == pytest.approx(TWO_PI)
    assert all(0 <= a.t0 < a.t1 - 1e-9 and a.t1 - a.t0 < TWO_PI for a in arcs)
    # середина дуг: верх (y > 0 в координатах по соглашению sin) и низ
    mids = sorted(arc_point(a, 0.5 * (a.t0 + a.t1))[1] for a in arcs)
    assert mids == pytest.approx([-1, 1])


def test_circle_touched_by_segment_end_stays_full():
    circle = Arc(0, (0, 0), 1.0, 0.0, TWO_PI, 0, Style(), 0)
    out = planarize([circle, S(1, (1, 0), (3, 0))], TOL, MIN)
    arcs = _by_stroke(out, 0)
    assert len(arcs) == 1 and arcs[0].is_full
    assert len(_by_stroke(out, 1)) == 1


def test_circle_tangent_segment_stays_full():
    circle = Arc(0, (0, 0), 1.0, 0.0, TWO_PI, 0, Style(), 0)
    out = planarize([circle, S(1, (-3, 1), (3, 1))], TOL, MIN)
    arcs = _by_stroke(out, 0)
    assert len(arcs) == 1 and arcs[0].is_full
    assert len(_by_stroke(out, 1)) == 2


def test_circle_with_three_segments_gives_k_arcs():
    circle = Arc(0, (0, 0), 1.0, 0.0, TWO_PI, 0, Style(), 0)
    spokes = [
        S(i + 1, (0, 0), (2 * math.cos(a), 2 * math.sin(a))) for i, a in enumerate([0.5, 2.5, 4.5])
    ]
    out = planarize([circle, *spokes], TOL, MIN)
    arcs = _by_stroke(out, 0)
    assert len(arcs) == 3
    assert sum(a.t1 - a.t0 for a in arcs) == pytest.approx(TWO_PI)
    assert all(a.t1 - a.t0 < TWO_PI and 0 <= a.t0 < TWO_PI for a in arcs)


def test_partial_arc_cut_by_segment():
    arc = Arc(0, (0, 0), 2.0, 0.0, math.pi, 0, Style(), 0)  # верхняя в формулах (sin >= 0)
    cut = S(1, (0, -3), (0, 3))
    miss = S(2, (-3, -1), (3, -1))  # не пересекает дугу
    out = planarize([arc, cut, miss], TOL, MIN)
    arcs = _by_stroke(out, 0)
    assert len(arcs) == 2
    assert sorted((round(a.t0, 6), round(a.t1, 6)) for a in arcs) == [
        (0.0, round(math.pi / 2, 6)),
        (round(math.pi / 2, 6), round(math.pi, 6)),
    ]
    assert len(_by_stroke(out, 1)) == 3  # разрезы в (0, 2) дугой и в (0, -1) прямой miss
    assert len(_by_stroke(out, 2)) == 2


def test_arc_range_respected():
    arc = Arc(0, (0, 0), 2.0, 0.0, math.pi / 2, 0, Style(), 0)  # четверть, x>0, y>0
    seg = S(1, (-3, -1), (3, -1))  # пересекает окружность, но не дугу
    out = planarize([arc, seg], TOL, MIN)
    assert len(out) == 2


def test_arc_arc_intersection():
    a = Arc(0, (0, 0), 2.0, 0.0, TWO_PI, 0, Style(), 0)
    b = Arc(1, (2, 0), 2.0, 0.0, TWO_PI, 1, Style(), 1)
    out = planarize([a, b], TOL, MIN)
    assert len(_by_stroke(out, 0)) == 2 and len(_by_stroke(out, 1)) == 2
    for p in _by_stroke(out, 0):
        assert p.t1 - p.t0 < TWO_PI


def test_short_pieces_dropped_and_zero_length():
    prims = [S(0, (0, 0), (10, 0)), S(1, (5, -10), (5, 10)), S(2, (3, 3), (3, 3))]
    out = planarize(prims, TOL, 6.0)
    # куски горизонтали по 5 < 6 выброшены, куски вертикали по 10 остались
    assert [p.stroke_id for p in out] == [1, 1]
    assert planarize([S(2, (3, 3), (3, 3))], TOL, MIN) == []


def test_cut_near_endpoint_ignored():
    # пересечение в 0.0005 от конца: кусок короче snap_tol не создается
    out = planarize([S(0, (0, 0), (10, 0)), S(1, (0.0005, -1), (0.0005, 1))], TOL, MIN)
    assert len(_by_stroke(out, 0)) == 1
    assert len(_by_stroke(out, 1)) == 2


def test_determinism_and_order():
    rng = np.random.default_rng(3)
    prims = []
    for i in range(300):
        p = rng.uniform(0, 50, 2)
        q = p + rng.uniform(-8, 8, 2)
        prims.append(S(i, tuple(p), tuple(q)))
    prims += [
        Arc(
            1000 + i,
            tuple(rng.uniform(0, 50, 2)),
            float(rng.uniform(1, 5)),
            0.0,
            TWO_PI,
            1000 + i,
            Style(),
            1000 + i,
        )
        for i in range(20)
    ]
    o1 = planarize(prims, TOL, MIN, 7)
    o2 = planarize(prims, TOL, MIN, 7)
    assert o1 == o2
    assert [p.id for p in o1] == list(range(7, 7 + len(o1)))
    strokes = [p.stroke_id for p in o1]
    assert strokes == sorted(strokes, key=lambda s: [q.id for q in prims].index(s))


def test_pieces_preserve_total_length():
    rng = np.random.default_rng(5)
    prims = []
    for i in range(200):
        p = rng.uniform(0, 30, 2)
        q = p + rng.uniform(-6, 6, 2)
        prims.append(S(i, tuple(p), tuple(q)))
    out = planarize(prims, TOL, 1e-9)
    for s in prims:
        L = math.dist(s.p0, s.p1)
        got = sum(math.dist(p.p0, p.p1) for p in out if p.stroke_id == s.id)
        assert got == pytest.approx(L, abs=1e-6)


def test_empty():
    assert planarize([], TOL, MIN) == []
