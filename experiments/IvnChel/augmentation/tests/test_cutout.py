"""Тесты cutout: квадраты не трогают образцы, закрытые знаки получают occluded или теряют рамку."""

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet  # noqa: E402
from cutout import cutout  # noqa: E402

W, H = 300, 200


def make_sheet():
    image = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(image)
    boxes = []
    for category, (x, y, w, h) in [
        (LEGEND_ITEM, (10, 10, 20, 20)),
        (OBJECT, (100, 100, 20, 20)),
        (OBJECT, (200, 50, 20, 20)),
    ]:
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=0)
        boxes.append(Box(category, (x, y, w, h), {"legend_id": "1"}))
    return image, Sheet("a.png", W, H, boxes)


def test_legend_never_covered_and_input_unchanged():
    image, sheet = make_sheet()
    before = image.copy()
    for seed in range(20):
        out, new = cutout(image, sheet, count=10, size=(20, 80), seed=seed)
        assert out.crop((10, 10, 30, 30)).getextrema() == (0, 0)
        assert [b.bbox for b in new.boxes if b.category == LEGEND_ITEM] == [(10, 10, 20, 20)]
    assert image.tobytes() == before.tobytes()


def test_fully_covered_object_removed():
    image, sheet = make_sheet()
    out, new = cutout(image, sheet, count=30, size=(150, 150), on_objects=1.0, seed=1)
    kept = [b for b in new.boxes if b.category == OBJECT]
    assert len(kept) < 2
    assert "removed=" in new.history[-1]


def test_partially_covered_gets_occluded_and_visible_matches():
    image, sheet = make_sheet()
    found = False
    for seed in range(200):
        out, new = cutout(image, sheet, count=1, size=(12, 12), on_objects=1.0, seed=seed)
        for box in new.boxes:
            if box.attributes.get("occluded"):
                found = True
                x, y, w, h = (round(v) for v in box.bbox)
                assert out.crop((x, y, x + w, y + h)).getextrema()[1] == 255  # есть белое
    assert found


def test_zero_count_changes_nothing_and_seed_repeatable():
    image, sheet = make_sheet()
    out, new = cutout(image, sheet, count=0)
    assert out.tobytes() == image.tobytes()
    a, _ = cutout(image, sheet, count=5, seed=3)
    b, _ = cutout(image, sheet, count=5, seed=3)
    assert a.tobytes() == b.tobytes()


def test_bad_parameters():
    image, sheet = make_sheet()
    with pytest.raises(ValueError):
        cutout(image, sheet, size=(0, 10))
    with pytest.raises(ValueError):
        cutout(image, sheet, max_covered=2)
