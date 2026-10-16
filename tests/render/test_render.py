"""Тесты рендеринга: палитра, SVG, PDF, PNG, HTML."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from symspot.core.types import Arc, Drawing, Result, Segment, Similarity, Style
from symspot.raster.mask import rle_decode
from symspot.render.html_report import render_html
from symspot.render.palette import PALETTE, color_for
from symspot.render.pdf_overlay import render_pdf
from symspot.render.png_overlay import render_png
from symspot.render.svg_overlay import render_svg
from tests.render.samples import make_raster_sample, make_vector_sample

SVG_NS = "{http://www.w3.org/2000/svg}"


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _load(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


# ---------------------------------------------------------------------------
# Палитра
# ---------------------------------------------------------------------------


def test_palette_size_and_distinct():
    assert len(PALETTE) >= 20
    assert len(set(PALETTE)) == len(PALETTE)
    arr = np.array(PALETTE, dtype=float)
    d = np.linalg.norm(arr[:, None] - arr[None], axis=2) + np.eye(len(arr)) * 1e9
    assert d.min() > 40
    # серых нет (серый занят фоном)
    assert (arr.max(axis=1) - arr.min(axis=1)).min() > 60


def test_color_for_sequential_ids_distinct():
    for pattern in ("sym{}", "L{}", "{}", "class_{}_a1"):
        cols = [color_for(pattern.format(i)) for i in range(20)]
        if pattern.endswith("a1"):  # номер не в конце: только детерминизм
            continue
        assert len(set(cols)) == 20, pattern


def test_color_for_stable_across_hash_seeds():
    code = (
        "from symspot.render.palette import color_for;"
        "print(color_for('valve'), color_for('sym7'), color_for('Клапан'))"
    )
    outs = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outs.append(r.stdout)
    assert outs[0] == outs[1]
    assert str(color_for("valve")) in outs[0]


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def test_svg_valid_and_groups(tmp_path):
    drawing, result = make_vector_sample()
    p = render_svg(drawing, result, tmp_path / "o.svg")
    root = _load(p)
    assert root.get("viewBox") == f"0 0 {drawing.width:g} {drawing.height:g}"
    groups = {
        g.get("id"): g for g in root.iter(f"{SVG_NS}g") if g.get("class", "").startswith("inst")
    }
    assert len(groups) == len(result.instances)
    for inst in result.instances:
        g = groups[f"inst-{inst.id}"]
        assert g.get("data-legend") == inst.legend_id
        assert len(list(g)) == len(inst.piece_ids)
        assert g.get("stroke") == _hex(color_for(inst.legend_id))


def test_svg_color_stable_per_legend(tmp_path):
    drawing, result = make_vector_sample()
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    for g in root.iter(f"{SVG_NS}g"):
        if g.get("class", "").startswith("inst"):
            assert g.get("stroke") == _hex(color_for(g.get("data-legend")))


def test_svg_background_gray_and_width(tmp_path):
    drawing, result = make_vector_sample()
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    bg = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "background")
    assert float(bg.get("opacity")) == pytest.approx(0.35)
    assert bg.get("stroke") == "#000"
    widths = [float(p.get("stroke-width")) for p in bg.iter(f"{SVG_NS}path")]
    assert min(widths) >= 0.3
    texts = [t.text for t in bg.iter(f"{SVG_NS}text")]
    assert "P-101" in texts


def _arc_midpoint(d: str) -> tuple[tuple[float, float], float]:
    """Середина дуги из атрибута d вида ``M x y A r r 0 large sweep x y`` (SVG F.6.5)."""
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", d)]
    x1, y1, rx, _ry, _rot, large, sweep, x2, y2 = nums
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    num = rx**2 * rx**2 - rx**2 * dy**2 - rx**2 * dx**2
    den = rx**2 * dy**2 + rx**2 * dx**2
    coef = math.sqrt(max(0.0, num / den)) * (-1 if large == sweep else 1)
    cxp, cyp = coef * dy, -coef * dx
    cx, cy = cxp + (x1 + x2) / 2, cyp + (y1 + y2) / 2
    a1 = math.atan2(y1 - cy, x1 - cx)
    a2 = math.atan2(y2 - cy, x2 - cx)
    if sweep == 1:
        while a2 < a1:
            a2 += 2 * math.pi
    else:
        while a2 > a1:
            a2 -= 2 * math.pi
    am = (a1 + a2) / 2
    return (cx + rx * math.cos(am), cy + rx * math.sin(am)), rx


@pytest.mark.parametrize(
    ("t0", "t1"),
    [(0.0, math.pi / 2), (math.pi / 2, 2 * math.pi), (-0.5, 1.2), (3.0, 3.0 + 4.5)],
)
def test_svg_partial_arc_passes_through_midpoint(tmp_path, t0, t1):
    c, r = (30.0, 40.0), 10.0
    arc = Arc(1, c, r, t0, t1, 0, Style(0.5))
    drawing = Drawing("vector", 100, 100, 1.0, primitives=[arc], normalized=True)
    from symspot.core.types import Instance

    inst = Instance("a", Similarity.identity(), 1.0, 1.0, [1], [], (20, 30, 40, 50), id=0)
    result = Result("x", [inst], {}, {})
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    g = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "inst-0")
    path = list(g)[0]
    assert path.tag == f"{SVG_NS}path"
    (mx, my), rr = _arc_midpoint(path.get("d"))
    tm = (t0 + t1) / 2
    assert rr == pytest.approx(r)
    assert mx == pytest.approx(c[0] + r * math.cos(tm), abs=2e-3)
    assert my == pytest.approx(c[1] + r * math.sin(tm), abs=2e-3)


def test_svg_full_arc_is_circle_and_segment_is_line(tmp_path):
    drawing, result = make_vector_sample()
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    g0 = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "inst-0")
    assert [e.tag for e in g0] == [f"{SVG_NS}circle", f"{SVG_NS}line"]


def test_svg_ambiguous_dashed(tmp_path):
    drawing, result = make_vector_sample()
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    groups = {g.get("id"): g for g in root.iter(f"{SVG_NS}g")}
    assert groups["inst-2"].get("stroke-dasharray")
    assert groups["inst-0"].get("stroke-dasharray") is None


def test_svg_empty_result(tmp_path):
    drawing, _ = make_vector_sample()
    result = Result("x", [], {}, {})
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    inst = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "instances")
    assert len(list(inst)) == 0


def test_svg_unknown_piece_ids_skipped(tmp_path):
    drawing, result = make_vector_sample()
    result.instances[0].piece_ids = [*result.instances[0].piece_ids, 999999]
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    g0 = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "inst-0")
    assert len(list(g0)) == 2


def test_svg_no_parents_uses_primitives(tmp_path):
    seg = Segment(1, (0, 0), (10, 10), 0, Style(0.5))
    drawing = Drawing("vector", 20, 20, 1.0, primitives=[seg], normalized=True)
    root = _load(render_svg(drawing, Result("x", [], {}, {}), tmp_path / "o.svg"))
    bg = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "background")
    assert len(list(bg.iter(f"{SVG_NS}path"))) == 1


def test_svg_raster_embeds_images(tmp_path):
    drawing, result = make_raster_sample()
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    imgs = list(root.iter(f"{SVG_NS}image"))
    assert len(imgs) == 2  # фон + маска первого экземпляра
    assert all(
        i.get("{http://www.w3.org/1999/xlink}href").startswith("data:image/png") for i in imgs
    )
    mask_img = imgs[1]
    assert (mask_img.get("x"), mask_img.get("y")) == ("18", "28")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, rotation: int = 0) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.draw_line((10, 10), (190, 10), width=1)
    page.draw_rect(pymupdf.Rect(20, 30, 60, 60), width=1)
    if rotation:
        page.set_rotation(rotation)
    doc.save(str(path))
    doc.close()


def _pdf_drawing(path: Path, w: float, h: float, p0, p1) -> tuple[Drawing, Result]:
    from symspot.core.types import Instance

    seg = Segment(1, p0, p1, 0, Style(0.5))
    drawing = Drawing("vector", w, h, 1.0, primitives=[seg], source_path=str(path), normalized=True)
    inst = Instance("q", Similarity.identity(), 1.0, 1.0, [1], [], (0, 0, 1, 1), id=0)
    return drawing, Result(str(path), [inst], {}, {})


def test_pdf_from_pdf_source_adds_drawings(tmp_path):
    src = tmp_path / "src.pdf"
    _make_pdf(src)
    drawing, result = _pdf_drawing(src, 200, 100, (30, 70), (150, 80))
    out = render_pdf(drawing, result, tmp_path / "out.pdf")
    with pymupdf.open(src) as a, pymupdf.open(out) as b:
        assert len(b) == 1
        assert len(b[0].get_drawings()) > len(a[0].get_drawings())
        col = tuple(c / 255 for c in color_for("q"))
        colored = [
            d
            for d in b[0].get_drawings()
            if d.get("color")
            and all(abs(x - y) < 0.01 for x, y in zip(d["color"], col, strict=True))
        ]
        assert colored
        r = colored[0]["rect"]
        assert (r.x0, r.y0, r.x1, r.y1) == pytest.approx((30, 70, 150, 80), abs=0.6)


def test_pdf_rotated_page_position(tmp_path):
    src = tmp_path / "rot.pdf"
    _make_pdf(src, rotation=90)
    with pymupdf.open(src) as d:
        pg = d[0]
        w, h = pg.rect.width, pg.rect.height
        rm = pg.rotation_matrix
    assert (w, h) == (100, 200)
    # отрезок в координатах отображаемой страницы
    p0, p1 = (20.0, 40.0), (70.0, 150.0)
    drawing, result = _pdf_drawing(src, w, h, p0, p1)
    out = render_pdf(drawing, result, tmp_path / "out.pdf")
    with pymupdf.open(out) as b:
        pg = b[0]
        assert pg.rotation == 90
        col = tuple(c / 255 for c in color_for("q"))
        found = [
            d
            for d in pg.get_drawings()
            if d.get("color")
            and all(abs(x - y) < 0.01 for x, y in zip(d["color"], col, strict=True))
        ]
        assert found
        items = found[0]["items"]
        assert items[0][0] == "l"
        a, b2 = items[0][1], items[0][2]
        # get_drawings отдает неповернутые координаты: переводим ожидание через rotation_matrix
        exp0 = pymupdf.Point(*p0) * ~rm
        exp1 = pymupdf.Point(*p1) * ~rm
        got = sorted([(a.x, a.y), (b2.x, b2.y)])
        exp = sorted([(exp0.x, exp0.y), (exp1.x, exp1.y)])
        assert got == pytest.approx(exp, abs=1e-3)
        # и обратно: через rotation_matrix получаем координаты чертежа
        back = sorted([tuple(pymupdf.Point(*g) * rm) for g in got])
        assert back == pytest.approx(sorted([p0, p1]), abs=1e-3)


def test_pdf_arc_and_ambiguous_do_not_fail(tmp_path):
    src = tmp_path / "src.pdf"
    _make_pdf(src)
    drawing, result = make_vector_sample()
    drawing.source_path = str(src)
    drawing.width, drawing.height = 200, 100
    out = render_pdf(drawing, result, tmp_path / "out.pdf")
    with pymupdf.open(out) as b:
        assert len(b) == 1


def test_pdf_non_pdf_source_via_cairosvg(tmp_path):
    drawing, result = make_vector_sample()
    out = render_pdf(drawing, result, tmp_path / "o.pdf")
    with pymupdf.open(out) as b:
        assert len(b) == 1
        assert b[0].rect.width == pytest.approx(drawing.width, abs=1)


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------


def test_png_vector_size_and_class_pixels(tmp_path):
    drawing, result = make_vector_sample()
    out = render_png(drawing, result, tmp_path / "o.png", scale=2.0)
    im = cv2.imread(str(out))
    assert im.shape[:2] == (int(drawing.height * 2), int(drawing.width * 2))
    r, g, b = color_for("circ_diag")
    near = (np.abs(im.astype(int) - np.array([b, g, r])).sum(axis=2) < 12).sum()
    assert near > 100


def test_png_raster_mask_pixels_opaque(tmp_path):
    drawing, result = make_raster_sample()
    out = render_png(drawing, result, tmp_path / "o.png")
    im = cv2.imread(str(out))
    assert im.shape == (80, 120, 3)
    mask, (ox, oy) = rle_decode(result.instances[0].mask_rle)
    r, g, b = color_for("ring")
    ys, xs = np.nonzero(mask)
    px = im[ys + oy, xs + ox]
    assert (px == np.array([b, g, r])).all()
    # фон осветлен: исходный 0 -> 127, 255 -> 254
    assert im[0, 0, 0] >= 250
    assert 120 <= im[40, 5, 0] <= 135  # линия 0 -> ~127


def test_png_empty_result(tmp_path):
    drawing, _ = make_vector_sample()
    out = render_png(drawing, Result("x", [], {}, {}), tmp_path / "o.png", scale=1.0)
    assert cv2.imread(str(out)).shape[:2] == (int(drawing.height), int(drawing.width))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_html_contains_table_ids_script_and_ambiguous(tmp_path):
    drawing, result = make_vector_sample()
    out = render_html(drawing, result, tmp_path / "r.html")
    s = out.read_text(encoding="utf-8")
    assert "<table" in s and "<script>" in s
    for inst in result.instances:
        assert f'id="inst-{inst.id}"' in s
        assert f'<tr data-id="{inst.id}"' in s
    assert "Ambiguous (1)" in s
    assert "wheel" in s and "pointerdown" in s
    assert "Окружность с диагональю" in s
    assert "search" in s  # тайминги
    assert "http://" not in s.replace("http://www.w3.org", "")  # без CDN
    assert "https://" not in s


def test_html_embedded_svg_is_valid_xml(tmp_path):
    drawing, result = make_vector_sample()
    s = render_html(drawing, result, tmp_path / "r.html").read_text(encoding="utf-8")
    m = re.search(r"<svg .*?</svg>", s, re.S)
    ET.fromstring(m.group(0))


def test_html_empty_result(tmp_path):
    drawing, _ = make_vector_sample()
    out = render_html(drawing, Result("x.pdf", [], {}, {}), tmp_path / "r.html")
    s = out.read_text(encoding="utf-8")
    assert "Экземпляры не найдены" in s and "Ambiguous (0)" in s


def test_html_raster(tmp_path):
    drawing, result = make_raster_sample()
    s = render_html(drawing, result, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "data:image/png;base64" in s and 'id="inst-1"' in s


def test_html_escapes_legend(tmp_path):
    drawing, result = make_vector_sample()
    result.instances[0].legend_id = 'a<b>"&'
    s = render_html(drawing, result, tmp_path / "r.html").read_text(encoding="utf-8")
    assert 'a<b>"&' not in s
    ET.fromstring(re.search(r"<svg .*?</svg>", s, re.S).group(0))


def test_duplicate_ids_get_unique_groups(tmp_path):
    drawing, result = make_vector_sample()
    for inst in result.instances:
        inst.id = -1
    root = _load(render_svg(drawing, result, tmp_path / "o.svg"))
    ids = [g.get("id") for g in root.iter(f"{SVG_NS}g") if g.get("class", "").startswith("inst")]
    assert len(ids) == len(set(ids)) == len(result.instances)
