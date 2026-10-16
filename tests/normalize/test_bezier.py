import math

import numpy as np
import pytest

from symspot.core.geometry_utils import arc_point
from symspot.core.types import DEFAULT_STYLE, Arc, ParentPath, PathItem, Segment, Style
from symspot.normalize.bezier import fit_circle, flatten_bezier, parents_to_strokes

K = 0.5522847498307936


def _parent(items, pid=0, style=DEFAULT_STYLE):
    return ParentPath(pid, list(items), np.zeros((0, 2)), style=style)


def _circle_items(cx, cy, r):
    quads = [
        [(cx + r, cy), (cx + r, cy + K * r), (cx + K * r, cy + r), (cx, cy + r)],
        [(cx, cy + r), (cx - K * r, cy + r), (cx - r, cy + K * r), (cx - r, cy)],
        [(cx - r, cy), (cx - r, cy - K * r), (cx - K * r, cy - r), (cx, cy - r)],
        [(cx, cy - r), (cx + K * r, cy - r), (cx + r, cy - K * r), (cx + r, cy)],
    ]
    return [PathItem("cubic", tuple(q)) for q in quads]


def _cubic_points(ctrl, n=400):
    t = np.linspace(0, 1, n)[:, None]
    c = np.asarray(ctrl, float)
    u = 1 - t
    return u**3 * c[0] + 3 * u * u * t * c[1] + 3 * u * t * t * c[2] + t**3 * c[3]


def _dist_to_polyline(pts, poly):
    best = np.full(len(pts), np.inf)
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        d = b - a
        ll = float(d @ d)
        t = np.clip(((pts - a) @ d) / ll, 0, 1) if ll > 0 else np.zeros(len(pts))
        foot = a + t[:, None] * d
        best = np.minimum(best, np.hypot(*(pts - foot).T))
    return best


def test_flatten_straight_cubic_is_two_points():
    poly = flatten_bezier(np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float), 0.01)
    assert poly.shape == (2, 2)
    np.testing.assert_allclose(poly[0], [0, 0])
    np.testing.assert_allclose(poly[-1], [3, 3])


@pytest.mark.parametrize("tol", [0.5, 0.05, 0.005])
def test_flatten_s_curve_within_tolerance(tol):
    ctrl = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], float)
    poly = flatten_bezier(ctrl, tol)
    assert len(poly) > 2
    np.testing.assert_allclose(poly[0], ctrl[0])
    np.testing.assert_allclose(poly[-1], ctrl[-1])
    assert _dist_to_polyline(_cubic_points(ctrl), poly).max() <= tol


def test_flatten_quad():
    ctrl = np.array([[0, 0], [5, 10], [10, 0]], float)
    poly = flatten_bezier(ctrl, 0.01)
    t = np.linspace(0, 1, 300)[:, None]
    pts = (1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t * t * ctrl[2]
    assert _dist_to_polyline(pts, poly).max() <= 0.01


def test_flatten_bad_input():
    with pytest.raises(ValueError):
        flatten_bezier(np.zeros((2, 2)), 0.1)
    with pytest.raises(ValueError):
        flatten_bezier(np.zeros((4, 2)), 0.0)


def test_fit_circle_exact_and_noisy():
    th = np.linspace(0.2, 2.5, 30)
    pts = np.stack([3 + 2 * np.cos(th), -1 + 2 * np.sin(th)], axis=1)
    (cx, cy), r, rms = fit_circle(pts)
    assert (cx, cy) == pytest.approx((3, -1), abs=1e-9)
    assert r == pytest.approx(2, abs=1e-9)
    assert rms < 1e-9
    noisy = pts + np.random.default_rng(0).normal(scale=0.01, size=pts.shape)
    (_, _), r2, rms2 = fit_circle(noisy)
    assert abs(r2 - 2) < 0.05
    assert 0.003 < rms2 < 0.03


def test_fit_circle_degenerate():
    line = np.stack([np.arange(6.0), 2 * np.arange(6.0)], axis=1)
    _, r, rms = fit_circle(line)
    assert math.isinf(r) and math.isinf(rms)
    _, r, rms = fit_circle(np.ones((5, 2)))
    assert math.isinf(r) and math.isinf(rms)
    _, r, rms = fit_circle(np.zeros((2, 2)))
    assert math.isinf(r)


def test_circle_from_four_beziers_is_one_full_arc():
    items = _circle_items(10.0, -3.0, 5.0)
    out = parents_to_strokes({7: _parent(items, pid=7)}, 0.01, 0.02, start_id=100)
    assert len(out) == 1
    a = out[0]
    assert isinstance(a, Arc)
    assert a.is_full and a.t0 == 0.0 and a.t1 == pytest.approx(2 * math.pi)
    assert a.r == pytest.approx(5.0, abs=5e-3)
    assert a.center[0] == pytest.approx(10.0, abs=5e-3)
    assert a.center[1] == pytest.approx(-3.0, abs=5e-3)
    assert a.id == 100 and a.stroke_id == 100 and a.parent_id == 7


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("start_deg", [0, 90, 270])
def test_quarter_arc_angles_and_midpoint(start_deg, reverse):
    r = 4.0
    a0 = math.radians(start_deg)
    a1 = a0 + math.pi / 2
    p0 = np.array([r * math.cos(a0), r * math.sin(a0)])
    p3 = np.array([r * math.cos(a1), r * math.sin(a1)])
    t0v = np.array([-math.sin(a0), math.cos(a0)])
    t1v = np.array([-math.sin(a1), math.cos(a1)])
    c1, c2 = p0 + K * r * t0v, p3 - K * r * t1v
    ctrl = [p0, c1, c2, p3]
    if reverse:
        ctrl = ctrl[::-1]
    path = _parent([PathItem("cubic", tuple(map(tuple, ctrl)))])
    out = parents_to_strokes({0: path}, 0.01, 0.02)
    assert len(out) == 1 and isinstance(out[0], Arc)
    arc = out[0]
    assert 0.0 <= arc.t0 < 2 * math.pi
    assert arc.t1 - arc.t0 == pytest.approx(math.pi / 2, abs=2e-3)
    mid = arc_point(arc, 0.5 * (arc.t0 + arc.t1))
    expect = r * np.array([math.cos(a0 + math.pi / 4), math.sin(a0 + math.pi / 4)])
    np.testing.assert_allclose(mid, expect, atol=2e-2)
    # концы дуги совпадают с концами кривой
    ends = np.array([arc_point(arc, arc.t0), arc_point(arc, arc.t1)])
    want = np.array([p0, p3])
    got_ok = np.allclose(ends, want, atol=2e-2) or np.allclose(ends[::-1], want, atol=2e-2)
    assert got_ok


def test_s_curve_becomes_segments():
    ctrl = [(0, 0), (4, 0), (0, 4), (4, 4)]
    out = parents_to_strokes({0: _parent([PathItem("cubic", tuple(ctrl))])}, 0.01, 0.01, 5)
    assert len(out) > 2 and all(isinstance(s, Segment) for s in out)
    assert [s.id for s in out] == list(range(5, 5 + len(out)))
    poly = np.array([out[0].p0] + [s.p1 for s in out])
    assert _dist_to_polyline(_cubic_points(ctrl), poly).max() <= 0.01 + 1e-9


def test_lines_degenerate_text_and_style():
    st = Style(stroke_width=2.0, color=(1, 0, 0))
    items = [
        PathItem("line", ((0, 0), (3, 0))),
        PathItem("line", ((3, 0), (3, 0))),  # нулевой
        PathItem("text", ()),
        PathItem("line", ((3, 0), (3, 4))),
    ]
    out = parents_to_strokes({4: _parent(items, pid=4, style=st)}, 0.01, 0.01)
    assert len(out) == 2
    assert all(s.parent_id == 4 and s.style == st and s.stroke_id == s.id for s in out)


def test_empty_and_straight_curve():
    assert parents_to_strokes({}, 0.01, 0.01) == []
    assert parents_to_strokes({0: _parent([])}, 0.01, 0.01) == []
    straight = PathItem("cubic", ((0, 0), (1, 1), (2, 2), (3, 3)))
    out = parents_to_strokes({0: _parent([straight])}, 0.01, 0.01)
    assert len(out) == 1 and isinstance(out[0], Segment)


def test_chain_not_circle_falls_back_to_individual_curves():
    # дуга четверти круга + прямая кубика подряд: цепочка не окружность, первая кривая дуга
    quarter = _circle_items(0, 0, 5)[0]
    line_like = PathItem("cubic", ((0, 5), (-1, 5), (-2, 5), (-3, 5)))
    out = parents_to_strokes({0: _parent([quarter, line_like])}, 0.01, 0.02)
    assert isinstance(out[0], Arc)
    assert isinstance(out[1], Segment)
    assert len(out) == 2
