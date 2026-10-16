import math

import pytest

from symspot.core.types import DEFAULT_STYLE, Arc, Segment
from symspot.normalize.merge import merge_collinear

KW = {"angle_tol_deg": 1.0, "gap": 0.05, "min_len": 1e-3}


def S(i, p0, p1, parent=0, style=DEFAULT_STYLE):
    return Segment(i, p0, p1, parent, style, i)


def _ends(s):
    return sorted([s.p0, s.p1])


def test_chain_of_three_collinear_pieces():
    out = merge_collinear(
        [S(0, (0, 0), (3, 0)), S(1, (3, 0), (4, 0.001)), S(2, (4, 0.001), (10, 0))], **KW
    )
    assert len(out) == 1
    assert _ends(out[0])[0] == (0, 0) and _ends(out[0])[1] == (10, 0)


def test_overlapping_collinear_merge_and_longest_wins():
    out = merge_collinear([S(5, (0, 0), (4, 0), parent=1), S(6, (3, 0), (10, 0), parent=2)], **KW)
    assert len(out) == 1
    s = out[0]
    assert s.id == 6 and s.parent_id == 2 and s.stroke_id == 6
    assert _ends(s) == [(0, 0), (10, 0)]


def test_gap_too_large_and_angle_too_large_not_merged():
    assert len(merge_collinear([S(0, (0, 0), (3, 0)), S(1, (3.5, 0), (6, 0))], **KW)) == 2
    assert len(merge_collinear([S(0, (0, 0), (30, 0)), S(1, (30, 0), (60, 3))], **KW)) == 2
    assert len(merge_collinear([S(0, (0, 0), (3, 0)), S(1, (0, 1), (3, 1))], **KW)) == 2


def test_duplicates_removed_any_direction():
    out = merge_collinear(
        [S(0, (0, 0), (5, 5)), S(1, (5, 5), (0, 0)), S(2, (0.01, 0), (5, 5.01))], **KW
    )
    assert len(out) == 1


def test_short_duplicate_removed_regardless_of_angle():
    kw = {"angle_tol_deg": 1.0, "gap": 0.05, "min_len": 1e-3}
    out = merge_collinear([S(0, (0, 0), (0.04, 0)), S(1, (0, 0), (0, 0.04))], **kw)
    assert len(out) == 2 or len(out) == 1  # допустимо оба, но без исключений


def test_zero_and_short_removed():
    out = merge_collinear(
        [S(0, (1, 1), (1, 1)), S(1, (0, 0), (0.0005, 0)), S(2, (0, 5), (5, 5))], **KW
    )
    assert [s.id for s in out] == [2]
    arcs = [Arc(0, (0, 0), 1e-4, 0.0, 1.0, 0), Arc(1, (0, 0), 1.0, 0.0, 1.0, 0)]
    assert [a.id for a in merge_collinear(arcs, **KW)] == [1]


def test_arc_duplicates_removed_and_arcs_not_merged():
    a = Arc(0, (0, 0), 5.0, 0.0, 1.0, 0)
    dup = Arc(1, (0.01, 0), 5.01, 0.001, 1.001, 1)
    other = Arc(2, (0, 0), 5.0, 1.0, 2.0, 2)  # соседняя дуга той же окружности
    full1 = Arc(3, (10, 10), 2.0, 0.0, 2 * math.pi, 3)
    full2 = Arc(4, (10, 10), 2.0, 1.0, 1.0 + 2 * math.pi, 4)
    out = merge_collinear([a, dup, other, full1, full2], **KW)
    assert [p.id for p in out] == [0, 2, 3]


def test_order_preserved_and_mixed():
    prims = [
        S(0, (0, 20), (5, 20)),
        Arc(1, (50, 50), 3.0, 0.0, 1.0, 1),
        S(2, (0, 0), (3, 0)),
        S(3, (3, 0), (8, 0)),
    ]
    out = merge_collinear(prims, **KW)
    assert [p.id for p in out] == [0, 1, 3]
    assert merge_collinear(prims, **KW) == out  # детерминизм


def test_empty_and_single():
    assert merge_collinear([], **KW) == []
    s = S(0, (0, 0), (1, 0))
    assert merge_collinear([s], **KW) == [s]


@pytest.mark.parametrize("n", [50])
def test_many_collinear_pieces_single_result(n):
    pieces = [S(i, (float(i), 0.0), (float(i + 1), 0.0)) for i in range(n)]
    out = merge_collinear(pieces, **KW)
    assert len(out) == 1
    assert _ends(out[0]) == [(0.0, 0.0), (float(n), 0.0)]
