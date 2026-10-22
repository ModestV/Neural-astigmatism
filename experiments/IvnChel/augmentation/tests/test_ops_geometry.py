"""Тесты геометрических операций: после поворота и масштаба рамка должна лежать на том же знаке."""

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet  # noqa: E402
from ops import flip, rotate_90, scale  # noqa: E402

W, H = 120, 80


def make_sheet():
    """Лист 120x80: чёрный прямоугольник-«знак» (не квадрат и не в центре) и рамка вокруг него."""
    image = Image.new("L", (W, H), 255)
    ImageDraw.Draw(image).rectangle([20, 10, 49, 24], fill=0)  # x 20..49, y 10..24 включительно
    legend = Box(LEGEND_ITEM, (20, 10, 30, 15), {"legend_id": "1", "rotation": 0.0})
    obj = Box(OBJECT, (20, 10, 30, 15), {"legend_id": "1", "rotation": 90.0})
    return image, Sheet("a.png", W, H, [legend, obj])


def assert_box_on_mark(image, box):
    """Внутри рамки темно, а чуть левее и правее рамки светло."""
    x, y, w, h = (round(v) for v in box.bbox)
    inner = image.crop((x + 2, y + 2, x + w - 2, y + h - 2)).resize((1, 1), Image.Resampling.BOX)
    assert inner.getpixel((0, 0)) < 40
    assert image.getpixel((min(image.width - 1, x + w + 4), y + h // 2)) > 215
    assert image.getpixel((max(0, x - 5), y + h // 2)) > 215


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_rotate_box_stays_on_mark(angle):
    image, sheet = make_sheet()
    new_image, new_sheet = rotate_90(image, sheet, angle)
    assert new_image.size == (new_sheet.width, new_sheet.height)
    assert (new_sheet.width, new_sheet.height) == ((H, W) if angle != 180 else (W, H))
    for box in new_sheet.boxes:
        assert_box_on_mark(new_image, box)
        x, y, w, h = box.bbox
        assert 0 <= x and 0 <= y and x + w <= new_sheet.width and y + h <= new_sheet.height


def test_rotate_90_swaps_width_and_height():
    image, sheet = make_sheet()
    _, new_sheet = rotate_90(image, sheet, 90)
    assert new_sheet.boxes[0].bbox[2:] == (15, 30)


def test_four_quarter_turns_return_original():
    image, sheet = make_sheet()
    for _ in range(4):
        image, sheet = rotate_90(image, sheet, 90)
    _, original = make_sheet()
    assert [b.bbox for b in sheet.boxes] == [b.bbox for b in original.boxes]
    assert (sheet.width, sheet.height) == (W, H)
    assert [b.attributes["rotation"] for b in sheet.boxes] == [0.0, 90.0]  # угол вернулся


def test_rotate_keeps_legend_link_and_does_not_touch_input():
    image, sheet = make_sheet()
    before = [b.bbox for b in sheet.boxes]
    _, new_sheet = rotate_90(image, sheet, 270)
    assert [b.legend_id for b in new_sheet.boxes] == ["1", "1"]
    assert [b.category for b in new_sheet.boxes] == [LEGEND_ITEM, OBJECT]
    assert [b.bbox for b in sheet.boxes] == before
    assert sheet.history == [] and new_sheet.history == ["rotate_90:270"]


def test_rotate_without_rotation_attribute_does_not_invent_one():
    image, sheet = make_sheet()
    del sheet.boxes[0].attributes["rotation"]
    _, new_sheet = rotate_90(image, sheet, 90)
    assert "rotation" not in new_sheet.boxes[0].attributes


def test_rotate_rejects_other_angles():
    image, sheet = make_sheet()
    with pytest.raises(ValueError):
        rotate_90(image, sheet, 45)


@pytest.mark.parametrize("factor", [0.5, 2.0, 1.37])
def test_scale_box_stays_on_mark(factor):
    image, sheet = make_sheet()
    new_image, new_sheet = scale(image, sheet, factor)
    assert new_image.size == (new_sheet.width, new_sheet.height)
    for box in new_sheet.boxes:
        assert_box_on_mark(new_image, box)
    assert new_sheet.history == [f"scale:{factor:g}"]


def test_scale_rejects_nonpositive_factor():
    image, sheet = make_sheet()
    with pytest.raises(ValueError):
        scale(image, sheet, 0)


@pytest.mark.parametrize("axis", ["h", "v"])
def test_flip_box_stays_on_mark_and_double_flip_is_identity(axis):
    image, sheet = make_sheet()
    flipped, flipped_sheet = flip(image, sheet, axis)
    for box in flipped_sheet.boxes:
        assert_box_on_mark(flipped, box)
    _, back = flip(flipped, flipped_sheet, axis)
    _, original = make_sheet()
    assert [b.bbox for b in back.boxes] == [b.bbox for b in original.boxes]
    assert [b.attributes["rotation"] for b in back.boxes] == [0.0, 90.0]
    assert flipped_sheet.history == [f"flip:{axis}"]


def test_flip_rotation_attribute():
    image, sheet = make_sheet()  # углы 0 и 90
    _, horizontal = flip(image, sheet, "h")
    assert [b.attributes["rotation"] for b in horizontal.boxes] == [180.0, 90.0]
    _, vertical = flip(image, sheet, "v")
    assert [b.attributes["rotation"] for b in vertical.boxes] == [0.0, 270.0]


def test_flip_rejects_other_axis():
    image, sheet = make_sheet()
    with pytest.raises(ValueError):
        flip(image, sheet, "x")
