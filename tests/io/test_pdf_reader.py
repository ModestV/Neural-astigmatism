from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from symspot.core.types import Drawing, TextPrim
from symspot.io.pdf_reader import CURVE_SAMPLES, read_pdf

TOL = 1e-3


def _assert_pts(actual, expected) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=TOL
    )


def _make_pdf(path: Path, draw, width: float = 300, height: float = 200, rotation: int = 0) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    draw(page)
    if rotation:
        page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    return path


def _basic(page: pymupdf.Page) -> None:
    page.draw_line((10, 10), (100, 20))
    page.draw_bezier((10, 50), (40, 10), (70, 90), (100, 50))
    page.draw_rect(pymupdf.Rect(120, 20, 200, 80))
    page.insert_text((20, 150), "Hello", fontsize=12)


def _ids(d: Drawing) -> list[int]:
    return sorted([*d.parents, *(p.id for p in d.primitives)])


def test_line_curve_rect_text(tmp_path: Path) -> None:
    d = read_pdf(_make_pdf(tmp_path / "a.pdf", _basic))
    assert d.kind == "vector"
    assert (d.width, d.height) == (300.0, 200.0)
    assert d.units_per_px == 1.0 and d.normalized is False and d.page == 0
    assert d.source_path.endswith("a.pdf")
    # 3 пути + 1 родитель текста
    assert len(d.parents) == 4
    paths = [p for p in d.parents.values() if p.items[0].kind != "text"]
    kinds = [[it.kind for it in p.items] for p in paths]
    assert kinds == [["line"], ["cubic"], ["line"] * 4]

    line, curve, rect = paths
    _assert_pts(line.items[0].points, ((10, 10), (100, 20)))
    assert line.closed is False and line.source_ref == "pdf:0" and line.group == 0
    _assert_pts(curve.items[0].points, ((10, 50), (40, 10), (70, 90), (100, 50)))
    assert curve.polyline.shape == (CURVE_SAMPLES, 2)
    assert curve.polyline[0] == pytest.approx((10, 50), abs=TOL)
    assert curve.polyline[-1] == pytest.approx((100, 50), abs=TOL)
    assert rect.closed is True and rect.fill is False
    assert rect.polyline.shape == (5, 2)
    assert rect.bbox == pytest.approx((120, 20, 200, 80), abs=TOL)
    assert line.style.stroke_width == pytest.approx(1.0)
    assert line.style.color == (0.0, 0.0, 0.0)

    assert len(d.primitives) == 1
    t = d.primitives[0]
    assert isinstance(t, TextPrim)
    assert t.text == "Hello"
    assert t.bbox[0] == pytest.approx(20, abs=TOL)
    assert t.bbox[1] < 150 < t.bbox[3]
    assert 20 < t.bbox[2] < 70
    assert t.angle == pytest.approx(0.0, abs=1e-9)
    par = d.parents[t.parent_id]
    assert [it.kind for it in par.items] == ["text"] and par.polyline.shape == (0, 2)


def test_ids_consecutive_and_unique(tmp_path: Path) -> None:
    d = read_pdf(_make_pdf(tmp_path / "a.pdf", _basic))
    assert _ids(d) == list(range(len(d.parents) + len(d.primitives)))
    assert all(k == p.id for k, p in d.parents.items())


def test_rotated_page(tmp_path: Path) -> None:
    d = read_pdf(_make_pdf(tmp_path / "r.pdf", _basic, rotation=90))
    # страница 300x200 повернута на 90: отображаемая 200x300
    assert (d.width, d.height) == (200.0, 300.0)
    line = next(p for p in d.parents.values() if p.items[0].kind == "line")
    # (x, y) -> (W_disp - y, x) для поворота на 90 по часовой, W_disp = 200
    _assert_pts(line.items[0].points[0], (190, 10))
    _assert_pts(line.items[0].points[1], (180, 100))
    t = d.primitives[0]
    # текст повернут вместе со страницей: направление базовой линии вниз
    assert t.angle == pytest.approx(math.pi / 2, abs=1e-6)
    x0, y0, x1, y1 = t.bbox
    # исходный origin (20, 150) -> (200 - 150, 20) = (50, 20)
    assert x0 < 50 < x1 and y0 <= 20 + 1e-6
    assert y1 - y0 > x1 - x0


def test_two_subpaths_one_group(tmp_path: Path) -> None:
    def draw(page: pymupdf.Page) -> None:
        sh = page.new_shape()
        sh.draw_line((10, 10), (50, 10))
        sh.draw_line((50, 10), (50, 40))  # продолжение первого
        sh.draw_line((100, 100), (150, 100))  # разрыв: новый подпуть
        sh.finish(color=(1, 0, 0), width=2, closePath=False)
        sh.commit()

    d = read_pdf(_make_pdf(tmp_path / "s.pdf", draw))
    assert len(d.parents) == 2
    a, b = d.parents[0], d.parents[1]
    assert a.group == b.group == 0
    assert len(a.items) == 2 and len(b.items) == 1
    assert a.polyline.shape == (3, 2)
    assert a.style.stroke_width == pytest.approx(2.0)
    assert a.style.color == (1.0, 0.0, 0.0)
    assert not a.closed and not b.closed


def test_closed_polyline_and_fill_only(tmp_path: Path) -> None:
    def draw(page: pymupdf.Page) -> None:
        page.draw_polyline([(10, 10), (60, 10), (35, 50)], color=(0, 0, 1), closePath=True)
        page.draw_rect(pymupdf.Rect(100, 100, 140, 140), color=None, fill=(0, 1, 0))

    d = read_pdf(_make_pdf(tmp_path / "f.pdf", draw))
    tri, filled = d.parents[0], d.parents[1]
    assert tri.closed is True and tri.fill is False
    assert tri.polyline[0] == pytest.approx(tri.polyline[-1], abs=TOL)
    assert filled.fill is True and filled.closed is True
    assert filled.style.color == (0.0, 1.0, 0.0)


def test_empty_page(tmp_path: Path) -> None:
    d = read_pdf(_make_pdf(tmp_path / "e.pdf", lambda page: None))
    assert d.parents == {} and d.primitives == []
    assert (d.width, d.height) == (300.0, 200.0)


def test_missing_file_and_bad_page(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_pdf(tmp_path / "nope.pdf")
    path = _make_pdf(tmp_path / "a.pdf", _basic)
    with pytest.raises(ValueError):
        read_pdf(path, page=5)
