"""Тесты операций, меняющих только вид картинки: рамки не должны сдвинуться."""

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import photometric as ph  # noqa: E402
from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet  # noqa: E402

W, H = 200, 120


def make_sheet(mode="L"):
    """Белый лист с чёрным знаком-прямоугольником и тонкой линией, рамка вокруг знака."""
    image = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 30, 79, 54], fill=0)  # знак
    draw.line([(100, 90), (180, 90)], fill=0, width=1)  # тонкая линия
    boxes = [
        Box(LEGEND_ITEM, (40, 30, 40, 25), {"legend_id": "1"}),
        Box(OBJECT, (40, 30, 40, 25), {"legend_id": "1", "rotation": 0.0}),
    ]
    return image.convert(mode), Sheet("a.png", W, H, boxes)


def dark_pixels(image):
    return sum(image.convert("L").histogram()[:128])


ALL_OPS = [
    lambda i, s: ph.brightness_contrast(i, s, 1.2, 0.8),
    lambda i, s: ph.add_noise(i, s, 15, seed=1),
    lambda i, s: ph.blur(i, s, 1.0),
    lambda i, s: ph.jpeg_compress(i, s, 50),
    lambda i, s: ph.line_thickness(i, s, 1),
    lambda i, s: ph.line_thickness(i, s, -1),
    lambda i, s: ph.invert(i, s),
    lambda i, s: ph.threshold(i, s, 180),
    lambda i, s: ph.stains(i, s, 10, (2, 5), seed=3),
]


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("op", ALL_OPS)
def test_boxes_size_and_mode_unchanged(op, mode):
    image, sheet = make_sheet(mode)
    before = [(b.category, b.bbox, dict(b.attributes)) for b in sheet.boxes]
    new_image, new_sheet = op(image, sheet)
    assert new_image.size == image.size and new_image.mode == mode
    assert (new_sheet.width, new_sheet.height) == (W, H)
    assert [(b.category, b.bbox, b.attributes) for b in new_sheet.boxes] == before
    assert len(new_sheet.history) == 1 and sheet.history == []  # исходный лист не тронут


def test_other_modes_converted_to_rgb():
    image, sheet = make_sheet()
    new_image, _ = ph.blur(image.convert("P"), sheet, 0.5)
    assert new_image.mode == "RGB"


def test_brightness_and_contrast_direction():
    image, sheet = make_sheet()
    brighter, _ = ph.brightness_contrast(image.point(lambda v: v // 2), sheet, brightness=1.5)
    assert ImageStat.Stat(brighter).mean[0] > ImageStat.Stat(image.point(lambda v: v // 2)).mean[0]
    flat, _ = ph.brightness_contrast(image, sheet, contrast=0.3)
    assert ImageStat.Stat(flat).stddev[0] < ImageStat.Stat(image).stddev[0]


def test_noise_is_repeatable_and_zero_amplitude_is_identity():
    image, sheet = make_sheet()
    a, _ = ph.add_noise(image, sheet, 20, seed=5)
    b, _ = ph.add_noise(image, sheet, 20, seed=5)
    c, _ = ph.add_noise(image, sheet, 20, seed=6)
    assert ImageChops.difference(a, b).getbbox() is None
    assert ImageChops.difference(a, c).getbbox() is not None
    zero, _ = ph.add_noise(image, sheet, 0, seed=5)
    assert ImageChops.difference(zero, image).getbbox() is None


def test_blur_softens_sharp_edge():
    image, sheet = make_sheet()
    out, _ = ph.blur(image, sheet, 2.0)
    assert 0 < out.getpixel((39, 40)) < 255  # граница знака стала серой


def test_jpeg_changes_pixels_but_not_much():
    image, sheet = make_sheet()
    out, _ = ph.jpeg_compress(image, sheet, 20)
    assert ImageChops.difference(out, image).getbbox() is not None
    assert out.getpixel((60, 42)) < 60  # середина знака осталась тёмной


def test_line_thickness_changes_dark_area():
    image, sheet = make_sheet()
    thick, _ = ph.line_thickness(image, sheet, 1)
    thin, _ = ph.line_thickness(image, sheet, -1)
    assert dark_pixels(thick) > dark_pixels(image) > dark_pixels(thin)
    assert thin.getpixel((140, 90)) == 255  # линия толщиной 1 px исчезла
    same, _ = ph.line_thickness(image, sheet, 0)
    assert ImageChops.difference(same, image).getbbox() is None


def test_invert_twice_returns_original():
    image, sheet = make_sheet()
    once, s1 = ph.invert(image, sheet)
    twice, _ = ph.invert(once, s1)
    assert once.getpixel((0, 0)) == 0
    assert ImageChops.difference(twice, image).getbbox() is None


def test_threshold_gives_only_black_and_white():
    image, sheet = make_sheet()
    out, _ = ph.threshold(image.point(lambda v: v if v == 0 else 190), sheet, 200)
    histogram = out.histogram()
    assert sum(histogram[1:255]) == 0  # нет ничего, кроме 0 и 255


def test_stains_avoid_boxes_and_are_repeatable():
    image, sheet = make_sheet()
    out, _ = ph.stains(image, sheet, count=40, radius=(3, 6), seed=2)
    again, _ = ph.stains(image, sheet, count=40, radius=(3, 6), seed=2)
    assert ImageChops.difference(out, again).getbbox() is None
    assert ImageChops.difference(out, image).getbbox() is not None  # пятна нарисовались
    for box in sheet.boxes:
        x, y, w, h = (int(v) for v in box.bbox)
        area = (x - 2, y - 2, x + w + 2, y + h + 2)  # рамка с запасом
        assert ImageChops.difference(out.crop(area), image.crop(area)).getbbox() is None


@pytest.mark.parametrize(
    "call",
    [
        lambda i, s: ph.brightness_contrast(i, s, 0, 1),
        lambda i, s: ph.add_noise(i, s, -1),
        lambda i, s: ph.blur(i, s, -1),
        lambda i, s: ph.jpeg_compress(i, s, 0),
        lambda i, s: ph.threshold(i, s, 300),
        lambda i, s: ph.stains(i, s, 1, (0, 3)),
    ],
)
def test_bad_parameters_rejected(call):
    image, sheet = make_sheet()
    with pytest.raises(ValueError):
        call(image, sheet)
