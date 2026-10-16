from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from symspot.core.types import Drawing, ParentPath
from symspot.io.svg_reader import read_svg

TOL = 1e-3


def _assert_pts(actual, expected) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=TOL
    )


NS = 'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'


def _write(tmp_path: Path, body: str, attrs: str = 'width="300" height="200"') -> Path:
    p = tmp_path / "t.svg"
    p.write_text(f"<svg {NS} {attrs}>{body}</svg>", encoding="utf-8")
    return p


def _paths(d: Drawing) -> list[ParentPath]:
    return [p for p in d.parents.values() if p.items[0].kind != "text"]


def test_viewbox_differs_from_size(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '<path id="a" d="M 10 20 L 100 50" stroke="red" stroke-width="2"/>',
        'width="300pt" height="200pt" viewBox="0 0 300 200"',
    )
    d = read_svg(p)
    assert (d.width, d.height) == (300.0, 200.0)
    assert d.kind == "vector" and d.units_per_px == 1.0 and d.normalized is False
    assert d.primitives == []
    (a,) = _paths(d)
    assert a.items[0].kind == "line"
    _assert_pts(a.items[0].points, ((10, 20), (100, 50)))
    assert a.source_ref == "svg:#a" and a.group == 0
    assert a.style.color == (1.0, 0.0, 0.0)
    assert a.style.stroke_width == pytest.approx(2.0, abs=TOL)


def test_viewbox_with_origin_and_nonuniform_size(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        '<path d="M 10 20 L 310 220"/>',
        'width="600" height="800" viewBox="10 20 300 200"',
    )
    d = read_svg(p)
    assert (d.width, d.height) == (300.0, 200.0)
    (a,) = _paths(d)
    _assert_pts(a.items[0].points, ((10, 20), (310, 220)))
    assert a.source_ref == "svg:0"


def test_nested_group_transform_and_use(tmp_path: Path) -> None:
    body = (
        '<defs><path id="d" d="M0 0 L10 0 L10 10"/></defs>'
        '<g transform="translate(100 50)"><g transform="scale(2)">'
        '<path d="M 0 0 L 10 0"/>'
        '<use xlink:href="#d" x="5" y="5"/>'
        "</g></g>"
    )
    d = read_svg(_write(tmp_path, body, 'width="300" height="200" viewBox="0 0 300 200"'))
    a, b = _paths(d)
    _assert_pts(a.items[0].points, ((100, 50), (120, 50)))
    # use: перенос (5,5), затем scale(2) и translate(100,50)
    assert [it.kind for it in b.items] == ["line", "line"]
    _assert_pts(b.items[0].points, ((110, 60), (130, 60)))
    _assert_pts(b.items[1].points[1], (130, 80))
    assert a.group != b.group


def test_circle_is_cubics_on_circle(tmp_path: Path) -> None:
    d = read_svg(_write(tmp_path, '<circle cx="50" cy="60" r="10" fill="none" stroke="black"/>'))
    (c,) = _paths(d)
    assert c.closed is True
    assert len(c.items) >= 4 and all(it.kind == "cubic" for it in c.items)
    r = np.hypot(c.polyline[:, 0] - 50, c.polyline[:, 1] - 60)
    assert np.abs(r - 10).max() < 1e-3 * 10
    for it in c.items:  # стыковка items
        assert len(it.points) == 4


def test_arc_command(tmp_path: Path) -> None:
    d = read_svg(_write(tmp_path, '<path d="M 100 100 A 20 20 0 0 1 140 100"/>'))
    (a,) = _paths(d)
    assert all(it.kind == "cubic" for it in a.items)
    _assert_pts(a.items[0].points[0], (100, 100))
    _assert_pts(a.items[-1].points[-1], (140, 100))
    r = np.hypot(a.polyline[:, 0] - 120, a.polyline[:, 1] - 100)
    assert np.abs(r - 20).max() < 1e-2
    assert a.closed is False
    # sweep=1 -> по часовой на экране (ось Y вниз): дуга проходит через верх (y < 100)
    assert a.polyline[:, 1].min() == pytest.approx(80, abs=1e-2)


def test_multiple_subpaths_and_close(tmp_path: Path) -> None:
    body = '<path id="m" d="M 0 0 L 10 0 M 20 0 L 30 0 L 30 10 Z" fill="none" stroke="black"/>'
    d = read_svg(_write(tmp_path, body))
    a, b = _paths(d)
    assert a.group == b.group == 0
    assert a.source_ref == b.source_ref == "svg:#m"
    assert a.closed is False and len(a.items) == 1
    assert b.closed is True and len(b.items) == 3  # 2 line + замыкающая
    _assert_pts(b.items[-1].points, ((30, 10), (20, 0)))
    assert b.fill is False


def test_shapes_rect_polyline_quad(tmp_path: Path) -> None:
    body = (
        '<rect x="10" y="20" width="30" height="40" fill="blue"/>'
        '<polyline points="0,0 5,5 10,0" fill="none" stroke="black"/>'
        '<path d="M 0 0 Q 5 10 10 0" fill="none" stroke="black"/>'
    )
    d = read_svg(_write(tmp_path, body))
    r, pl, q = _paths(d)
    assert r.closed and len(r.items) == 4 and r.fill is True
    assert r.bbox == pytest.approx((10, 20, 40, 60), abs=TOL)
    assert r.style.color == (0.0, 0.0, 1.0)
    assert not pl.closed and len(pl.items) == 2 and pl.fill is False
    assert [it.kind for it in q.items] == ["quad"]
    assert q.polyline.shape == (16, 2)


def test_text_with_rotation(tmp_path: Path) -> None:
    body = '<text id="t" x="10" y="20" font-size="12" transform="rotate(90 10 20)">Hello</text>'
    d = read_svg(_write(tmp_path, body))
    (t,) = d.primitives
    assert t.text == "Hello"
    assert t.angle == pytest.approx(math.pi / 2, abs=1e-6)
    # локальная рамка x 10..46, y 10.4..22.4; поворот на 90 вокруг (10, 20):
    # (x, y) -> (10 - (y - 20), 20 + (x - 10))
    assert t.bbox == pytest.approx((7.6, 20.0, 19.6, 56.0), abs=1e-6)
    par = d.parents[t.parent_id]
    assert [it.kind for it in par.items] == ["text"] and par.source_ref == "svg:#t"


def test_text_anchor_middle_and_plain(tmp_path: Path) -> None:
    body = '<text x="100" y="50" font-size="10" text-anchor="middle">ABCD</text>'
    d = read_svg(_write(tmp_path, body))
    (t,) = d.primitives
    assert t.angle == pytest.approx(0.0, abs=1e-9)
    assert t.bbox == pytest.approx((100 - 12, 50 - 8, 100 + 12, 50 + 2), abs=1e-6)


def test_empty_svg(tmp_path: Path) -> None:
    d = read_svg(_write(tmp_path, "", 'width="300" height="200" viewBox="0 0 300 200"'))
    assert d.parents == {} and d.primitives == []
    assert (d.width, d.height) == (300.0, 200.0)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_svg(tmp_path / "nope.svg")


def test_ids_consecutive_and_unique(tmp_path: Path) -> None:
    body = (
        '<path d="M 0 0 L 1 1 M 5 5 L 6 6"/>'
        '<text x="1" y="2">A</text>'
        '<circle cx="5" cy="5" r="2"/>'
        '<text x="3" y="4">B</text>'
    )
    d = read_svg(_write(tmp_path, body))
    ids = sorted([*d.parents, *(p.id for p in d.primitives)])
    assert ids == list(range(len(d.parents) + len(d.primitives)))
    assert all(k == p.id for k, p in d.parents.items())
    assert {p.parent_id for p in d.primitives} <= set(d.parents)


@pytest.mark.slow
def test_30k_paths_speed(tmp_path: Path) -> None:
    n = 30000
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, size=(n, 4))
    body = "".join(
        f'<path d="M {a:.2f} {b:.2f} L {c:.2f} {e:.2f}" fill="none" stroke="black"/>'
        for a, b, c, e in xy
    )
    p = _write(tmp_path, body, 'width="1000" height="1000" viewBox="0 0 1000 1000"')
    t0 = time.perf_counter()
    d = read_svg(p)
    elapsed = time.perf_counter() - t0
    assert len(d.parents) == n
    assert elapsed < 10.0, f"чтение заняло {elapsed:.1f} с"
