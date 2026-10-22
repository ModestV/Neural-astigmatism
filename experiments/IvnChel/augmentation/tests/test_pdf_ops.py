"""Тесты векторных операций: после поворота и масштаба PDF рамка должна лежать на том же знаке."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pymupdf = pytest.importorskip("pymupdf")

from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet  # noqa: E402
from pdf_ops import (  # noqa: E402
    DPI, cutout_pdf, page_pixel_size, restyle_pdf, tile_pdf, transform_pdf,
)
from pdf_style import _tokens  # noqa: E402

K = DPI / 72
PT_W, PT_H = 144, 72  # страница 400x200 px при 200 dpi
MARK = pymupdf.Rect(36, 18, 72, 36)  # чёрный «знак» 36x18 пунктов, не по центру


def make_pdf(path: Path, rotation: int) -> Sheet:
    """PDF с одним чёрным прямоугольником; у страницы свой /Rotate (как у 02, 04, 09)."""
    doc = pymupdf.open()
    page = doc.new_page(width=PT_W, height=PT_H)
    page.draw_rect(MARK, color=None, fill=(0, 0, 0))
    page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as check:
        shown = check[0].rect  # с учётом /Rotate
        width, height = round(shown.width * K), round(shown.height * K)
        # Знак в системе «как показано»: считаем рендером, а не формулами, чтобы тест был честным.
        pix = check[0].get_pixmap(dpi=DPI, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    bbox = image.point(lambda v: 255 if v < 128 else 0).getbbox()
    x0, y0, x1, y1 = bbox
    box = (x0, y0, x1 - x0, y1 - y0)
    return Sheet("a.pdf", width, height, [
        Box(LEGEND_ITEM, box, {"legend_id": "1", "rotation": 0.0}),
        Box(OBJECT, box, {"legend_id": "1", "rotation": 90.0}),
    ])


def render(path: Path) -> Image.Image:
    with pymupdf.open(path) as doc:
        data = doc[0].get_pixmap(dpi=DPI, alpha=False).tobytes("png")
    return Image.open(io.BytesIO(data)).convert("L")


def assert_box_on_mark(image: Image.Image, box: Box) -> None:
    x, y, w, h = (round(v) for v in box.bbox)
    inner = image.crop((x + 3, y + 3, x + w - 3, y + h - 3)).resize((1, 1), Image.Resampling.BOX)
    assert inner.getpixel((0, 0)) < 40
    # Рамка плотно облегает знак: сразу за её краями светло.
    for probe in ((x - 4, y + h // 2), (x + w + 3, y + h // 2), (x + w // 2, y - 4),
                  (x + w // 2, y + h + 3)):
        assert image.getpixel(probe) > 215


@pytest.mark.parametrize("source_rotation", [0, 270])
@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("factor", [1.0, 0.5, 2.0])
def test_boxes_follow_the_mark(tmp_path, source_rotation, angle, factor):
    sheet = make_pdf(tmp_path / "src.pdf", source_rotation)
    out = tmp_path / "out.pdf"
    new = transform_pdf(tmp_path / "src.pdf", sheet, out, angle, factor)
    image = render(out)
    assert image.size == (new.width, new.height)
    for box in new.boxes:
        assert_box_on_mark(image, box)


def test_page_size_and_rotation_attribute(tmp_path):
    sheet = make_pdf(tmp_path / "src.pdf", 0)
    new = transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "o.pdf", 90, 1.0)
    assert (new.width, new.height) == (sheet.height, sheet.width)
    legend, obj = new.boxes
    # Как в растровом ops.rotate_90: угол уменьшается на поворот листа против часовой стрелки.
    assert legend.attributes["rotation"] == 270.0
    assert obj.attributes["rotation"] == 0.0
    assert new.history == ["rotate_90:90"]


def test_scale_history_and_source_untouched(tmp_path):
    sheet = make_pdf(tmp_path / "src.pdf", 0)
    before = list(sheet.history)
    new = transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "o.pdf", 0, 0.5)
    assert new.history == ["pdf_scale:0.5"]
    assert sheet.history == before and sheet.width == 400


def test_rejects_wrong_markup_size(tmp_path):
    sheet = make_pdf(tmp_path / "src.pdf", 0)
    bad = Sheet("a.pdf", sheet.width + 10, sheet.height, sheet.boxes)
    with pytest.raises(ValueError):
        transform_pdf(tmp_path / "src.pdf", bad, tmp_path / "o.pdf", 0, 1.0)


def test_rejects_bad_arguments(tmp_path):
    sheet = make_pdf(tmp_path / "src.pdf", 0)
    with pytest.raises(ValueError):
        transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "o.pdf", 45, 1.0)
    with pytest.raises(ValueError):
        transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "o.pdf", 0, 0)


@pytest.mark.parametrize("source_rotation", [0, 270])
@pytest.mark.parametrize("angle", [0, 90, 270])
def test_cutout_squares_match_boxes(tmp_path, source_rotation, angle):
    """Квадрат стоит там, где его считает разметка: целый знак на месте, закрытый — без рамки."""
    sheet = make_pdf(tmp_path / "src.pdf", source_rotation)
    sheet.boxes = [b for b in sheet.boxes if b.category == OBJECT]  # образец не даёт ставить квадраты
    changed = 0
    for seed in range(10):
        out = tmp_path / f"o{seed}.pdf"
        new = transform_pdf(tmp_path / "src.pdf", sheet, out, angle, 1.0, 3, (15, 40), seed)
        image = render(out)
        assert "cutout" in new.history[-1]
        reference = transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "ref.pdf", angle, 1.0)
        ref_box = reference.boxes[0]
        x, y, w, h = (round(v) for v in ref_box.bbox)
        dark = image.crop((x, y, x + w, y + h)).point(lambda v: 255 if v < 128 else 0)
        share = dark.histogram()[255] / (w * h)
        if not new.boxes:  # рамку убрали: знак закрыт больше чем на треть
            changed += 1
            assert share < 0.7
        elif new.boxes[0].attributes.get("occluded"):
            changed += 1
        else:
            assert share > 0.99  # знак цел
    assert changed > 0  # тест не пустой: хоть раз квадрат на знак попал


def test_cutout_zero_changes_nothing(tmp_path):
    sheet = make_pdf(tmp_path / "src.pdf", 0)
    a = transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "a.pdf", 90, 1.0)
    b = transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "b.pdf", 90, 1.0, 0)
    assert a.boxes == b.boxes and a.history == b.history
    assert render(tmp_path / "a.pdf").tobytes() == render(tmp_path / "b.pdf").tobytes()


def make_two_marks(path: Path, rotation: int) -> Sheet:
    """PDF с образцом (слева, legend_item) и экземпляром (справа, object); рамки — по рендеру."""
    doc = pymupdf.open()
    page = doc.new_page(width=PT_W, height=PT_H)
    page.draw_rect(pymupdf.Rect(10, 10, 30, 22), color=None, fill=(0, 0, 0))
    page.draw_rect(pymupdf.Rect(100, 40, 130, 60), color=None, fill=(0, 0, 0))
    page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    image = render(path)
    mask = image.point(lambda v: 255 if v < 128 else 0)
    # Две отдельные тёмные области: делим лист пополам по длинной стороне.
    half = image.width // 2 if image.width >= image.height else None
    boxes = []
    for part in (0, 1):
        if half is not None:
            area = (0, 0, half, image.height) if part == 0 else (half, 0, image.width, image.height)
        else:
            half_h = image.height // 2
            area = (0, 0, image.width, half_h) if part == 0 else (0, half_h, image.width, image.height)
        x0, y0, x1, y1 = mask.crop(area).getbbox()
        boxes.append((x0 + area[0], y0 + area[1], x1 - x0, y1 - y0))
    with pymupdf.open(path) as check:
        width, height = page_pixel_size(check[0])
    # какой из двух знаков образец, зависит от поворота: образец — тот, что меньше по площади
    small, big = sorted(boxes, key=lambda b: b[2] * b[3])
    return Sheet("a.pdf", width, height, [
        Box(LEGEND_ITEM, small, {"legend_id": "1", "rotation": 0.0}),
        Box(OBJECT, big, {"legend_id": "1", "rotation": 0.0}),
    ])


@pytest.mark.parametrize("source_rotation", [0, 270])
@pytest.mark.parametrize("angle", [0, 90, 270])
@pytest.mark.parametrize("mode", ["keep", "with-legend"])
def test_tiles_match_boxes(tmp_path, source_rotation, angle, mode):
    sheet = make_two_marks(tmp_path / "src.pdf", source_rotation)
    whole = transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "w.pdf", angle, 1.0)
    tiles = tile_pdf(tmp_path / "w.pdf", whole, tmp_path / "tiles", (160, 130), 20,
                     legend_mode=mode)
    assert tiles
    found_object = False
    for tile in tiles:
        path = tmp_path / "tiles" / tile.file_name
        image = render(path)
        assert image.size == (tile.width, tile.height)
        for box in tile.boxes:
            if box.attributes.get("truncated"):
                continue
            assert_box_on_mark(image, box)
            found_object |= box.category == OBJECT
    assert found_object
    if mode == "with-legend":
        assert any(b.attributes.get("pasted") for t in tiles for b in t.boxes)
        assert all(b.attributes["legend_in_tile"] for t in tiles for b in t.boxes
                   if b.category == OBJECT and not b.attributes.get("truncated"))


def test_tile_pdf_rejects_wrong_markup(tmp_path):
    sheet = make_two_marks(tmp_path / "src.pdf", 0)
    bad = Sheet("a.pdf", sheet.width + 5, sheet.height, sheet.boxes)
    with pytest.raises(ValueError):
        tile_pdf(tmp_path / "src.pdf", bad, tmp_path / "t", (100, 100))
    with pytest.raises(ValueError):
        tile_pdf(tmp_path / "src.pdf", sheet, tmp_path / "t", (100, 100), legend_mode="nope")


def test_cutout_pdf_on_finished_page(tmp_path):
    sheet = make_pdf(tmp_path / "src.pdf", 270)
    sheet.boxes = [b for b in sheet.boxes if b.category == OBJECT]
    new = cutout_pdf(tmp_path / "src.pdf", sheet, tmp_path / "c.pdf", 3, (15, 40), seed=1)
    assert new.file_name == "c.pdf" and "cutout" in new.history[-1]
    assert render(tmp_path / "c.pdf").size == (sheet.width, sheet.height)
    bad = Sheet("a.pdf", sheet.width + 1, sheet.height, sheet.boxes)
    with pytest.raises(ValueError):
        cutout_pdf(tmp_path / "src.pdf", bad, tmp_path / "c2.pdf", 1)


def make_lines_pdf(path: Path, rotation: int = 0) -> Sheet:
    """PDF с горизонтальной линией толщиной 1 пт и вертикальной толщиной 2 пт (без заливок)."""
    doc = pymupdf.open()
    page = doc.new_page(width=PT_W, height=PT_H)
    page.draw_line((10, 36), (134, 36), color=(0, 0, 0), width=1)
    page.draw_line((100, 5), (100, 67), color=(0, 0, 0), width=2)
    page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as check:
        width, height = page_pixel_size(check[0])
    return Sheet("l.pdf", width, height, [Box(OBJECT, (0, 0, 10, 10), {"legend_id": "1"})])


def run_length(image: Image.Image, x: int, y: int, vertical: bool) -> int:
    """Длина тёмной полосы, проходящей через точку (x, y), вдоль выбранной оси."""
    dark = lambda px, py: image.getpixel((px, py)) < 128  # noqa: E731
    step = (0, 1) if vertical else (1, 0)
    a = 0
    while dark(x - step[0] * (a + 1), y - step[1] * (a + 1)):
        a += 1
    b = 0
    while dark(x + step[0] * (b + 1), y + step[1] * (b + 1)):
        b += 1
    return a + b + 1


@pytest.mark.parametrize("factor", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("delta", [2, 1, -1])
def test_thickness_changes_lines_by_pixels(tmp_path, factor, delta):
    """Линия меняется на 2*delta px листа при любом масштабе (поправка учитывает вложенные формы)."""
    sheet = make_lines_pdf(tmp_path / "src.pdf")
    scaled = transform_pdf(tmp_path / "src.pdf", sheet, tmp_path / "s.pdf", 0, factor)
    before = render(tmp_path / "s.pdf")
    restyle_pdf(tmp_path / "s.pdf", scaled, tmp_path / "t.pdf", thickness=delta)
    after = render(tmp_path / "t.pdf")
    # Горизонтальная линия: смотрим толщину по вертикали в точке вдали от вертикальной.
    x, y = round(30 * K * factor), round(36 * K * factor)
    w0, w1 = run_length(before, x, y, True), run_length(after, x, y, True)
    assert abs((w1 - w0) - 2 * delta) <= 1
    # Вертикальная линия (толщина 2 пт): то же по горизонтали.
    x, y = round(100 * K * factor), round(15 * K * factor)
    v0, v1 = run_length(before, x, y, False), run_length(after, x, y, False)
    assert abs((v1 - v0) - 2 * delta) <= 1


def test_thickness_keeps_page_geometry_and_boxes(tmp_path):
    sheet = make_lines_pdf(tmp_path / "src.pdf", 270)
    new = restyle_pdf(tmp_path / "src.pdf", sheet, tmp_path / "t.pdf", thickness=1)
    assert (new.width, new.height) == (sheet.width, sheet.height)
    assert new.boxes == sheet.boxes and new.history[-1] == "pdf_thickness:1"


def test_very_thin_lines_do_not_vanish(tmp_path):
    sheet = make_lines_pdf(tmp_path / "src.pdf")
    restyle_pdf(tmp_path / "src.pdf", sheet, tmp_path / "t.pdf", thickness=-20)
    image = render(tmp_path / "t.pdf")
    assert image.point(lambda v: 255 if v < 200 else 0).getbbox() is not None


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_invert_is_exact(tmp_path, rotation):
    from PIL import ImageChops, ImageOps

    sheet = make_two_marks(tmp_path / "src.pdf", rotation)
    restyle_pdf(tmp_path / "src.pdf", sheet, tmp_path / "i.pdf", invert=True)
    expected = ImageOps.invert(render(tmp_path / "src.pdf"))
    diff = ImageChops.difference(render(tmp_path / "i.pdf"), expected)
    assert diff.getbbox() is None


def test_thickness_then_invert_then_cutout_chain(tmp_path):
    sheet = make_lines_pdf(tmp_path / "src.pdf")
    step = restyle_pdf(tmp_path / "src.pdf", sheet, tmp_path / "a.pdf", 1, True)
    final = cutout_pdf(tmp_path / "a.pdf", step, tmp_path / "b.pdf", 2, (10, 20), seed=3)
    assert step.history[-2:] == ["pdf_thickness:1", "pdf_invert"]
    assert render(tmp_path / "b.pdf").size == (sheet.width, sheet.height)
    assert final.history[-1].startswith("cutout")


def test_restyle_rejects_bad_arguments(tmp_path):
    sheet = make_lines_pdf(tmp_path / "src.pdf")
    with pytest.raises(ValueError):
        restyle_pdf(tmp_path / "src.pdf", sheet, tmp_path / "o.pdf")
    bad = Sheet("l.pdf", sheet.width + 1, sheet.height, sheet.boxes)
    with pytest.raises(ValueError):
        restyle_pdf(tmp_path / "src.pdf", bad, tmp_path / "o.pdf", thickness=1)


def test_tokenizer_ignores_strings_and_inline_images():
    data = b"2 w (not 5 w op) Tj [(a) -3 (w)] TJ BI /W 1 /H 1 ID \x00w w\x01 EI 4 w"
    words = [t for kind, t, *_ in _tokens(data) if kind == "word"]
    assert words.count(b"w") == 2 and b"Tj" in words and b"TJ" in words
