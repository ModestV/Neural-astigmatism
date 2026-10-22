"""Тесты нарезки на плитки: рамки после нарезки должны лежать на тех же знаках."""

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coco_io import LEGEND_ITEM, OBJECT, Box, Sheet  # noqa: E402
from tiling import tile_positions, tile_sheet  # noqa: E402

W, H = 100, 60


def make_sheet(extra_legend=None):
    """Лист 100x60, плитки 50x30 без перекрытия дают четыре плитки.

    Образец (5,5), объект рядом (20,10), объект на границе плиток (45,20) шириной 10 (по 5 px в двух
    плитках), одинокий объект в правой нижней плитке (60,40). У всех legend_id = "1".
    """
    image = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(image)
    specs = [
        (LEGEND_ITEM, (5, 5, 10, 8)),
        (OBJECT, (20, 10, 10, 8)),
        (OBJECT, (45, 20, 10, 8)),
        (OBJECT, (60, 40, 10, 8)),
    ]
    if extra_legend:
        specs.append((LEGEND_ITEM, extra_legend))
    boxes = []
    for category, (x, y, w, h) in specs:
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=0)
        boxes.append(Box(category, (x, y, w, h), {"legend_id": "1"}))
    return image, Sheet("plan.jpg", W, H, boxes, source="s.json")


def assert_boxes_on_marks(tile_image, tile):
    """Рамка внутри плитки, и внутри неё тёмный знак."""
    assert tile_image.size == (tile.width, tile.height)
    for box in tile.boxes:
        x, y, w, h = box.bbox
        assert 0 <= x and 0 <= y and x + w <= tile.width and y + h <= tile.height
        inner = tile_image.crop((round(x), round(y), round(x + w), round(y + h)))
        assert inner.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)) < 20


def by_name(tiles):
    return {t.file_name: (i, t) for i, t in tiles}


def test_tile_positions_cover_edge():
    assert tile_positions(100, 40, 40) == [0, 40, 60]  # последняя прижата к краю
    assert tile_positions(100, 50, 50) == [0, 50]
    assert tile_positions(30, 50, 50) == [0]  # плитка больше листа
    assert tile_positions(100, 50, 25) == [0, 25, 50]


def test_keep_mode_boxes_stay_on_marks_and_flags():
    image, sheet = make_sheet()
    tiles = tile_sheet(image, sheet, (50, 30))
    assert len(tiles) == 3  # левая нижняя плитка пустая и пропускается
    for tile_image, tile in tiles:
        assert_boxes_on_marks(tile_image, tile)
    named = by_name(tiles)
    _, top_left = named["plan__tile_x0_y0.jpg"]
    flags = {b.bbox: b.attributes["legend_in_tile"] for b in top_left.boxes if b.category == OBJECT}
    assert flags == {(20, 10, 10, 8): True, (45, 20, 5, 8): True}
    _, bottom_right = named["plan__tile_x50_y30.jpg"]
    assert [b.attributes["legend_in_tile"] for b in bottom_right.boxes] == [False]
    assert all("legend_in_tile" not in b.attributes for b in top_left.boxes
               if b.category == LEGEND_ITEM)


def test_straddling_box_is_clipped_and_marked_truncated():
    image, sheet = make_sheet()
    named = by_name(tile_sheet(image, sheet, (50, 30)))
    _, left = named["plan__tile_x0_y0.jpg"]
    _, right = named["plan__tile_x50_y0.jpg"]
    cut_left = [b for b in left.boxes if b.attributes.get("truncated")]
    cut_right = [b for b in right.boxes if b.attributes.get("truncated")]
    assert [b.bbox for b in cut_left] == [(45, 20, 5, 8)]
    assert [b.bbox for b in cut_right] == [(0, 20, 5, 8)]
    assert right.boxes[0].attributes["legend_in_tile"] is False  # образец в другой плитке


def test_min_visible_drops_boxes_that_are_cut_too_much():
    image, sheet = make_sheet()
    named = by_name(tile_sheet(image, sheet, (50, 30), min_visible=0.6))
    assert "plan__tile_x50_y0.jpg" not in named  # в ней осталась бы только половинка объекта
    _, left = named["plan__tile_x0_y0.jpg"]
    assert [b.bbox for b in left.boxes if b.category == OBJECT] == [(20, 10, 10, 8)]


def test_require_mode_keeps_only_tiles_where_every_object_has_its_legend():
    image, sheet = make_sheet()
    tiles = tile_sheet(image, sheet, (50, 30), legend_mode="require")
    assert [t.file_name for _, t in tiles] == ["plan__tile_x0_y0.jpg"]
    assert all(b.attributes["legend_in_tile"] for _, t in tiles for b in t.boxes
               if b.category == OBJECT)


def test_cut_legend_item_is_not_counted_as_present():
    image, sheet = make_sheet(extra_legend=(45, 40, 10, 8))  # образец режется границей плиток
    named = by_name(tile_sheet(image, sheet, (50, 30)))
    assert "plan__tile_x0_y30.jpg" not in named  # половинка образца отброшена, плитка пустая
    _, bottom_right = named["plan__tile_x50_y30.jpg"]
    assert [b.category for b in bottom_right.boxes] == [OBJECT]  # вторая половинка тоже отброшена
    assert bottom_right.boxes[0].attributes["legend_in_tile"] is False


def test_empty_tiles_skipped_unless_requested():
    image, sheet = make_sheet()
    sheet.boxes = [b for b in sheet.boxes if b.bbox[0] < 30]  # остался образец и один объект
    assert len(tile_sheet(image, sheet, (50, 30))) == 1
    assert len(tile_sheet(image, sheet, (50, 30), keep_empty=True)) == 4


def test_overlap_and_input_untouched_and_history():
    image, sheet = make_sheet()
    before = [(b.bbox, dict(b.attributes)) for b in sheet.boxes]
    tiles = tile_sheet(image, sheet, (50, 30), overlap=10)
    assert len({t.file_name for _, t in tiles}) == len(tiles)  # имена уникальны
    for tile_image, tile in tiles:
        assert_boxes_on_marks(tile_image, tile)
        assert tile.history[-1].startswith("tile:")
        assert tile.source == "s.json"
    assert [(b.bbox, dict(b.attributes)) for b in sheet.boxes] == before
    assert sheet.history == []


def test_tile_larger_than_sheet_gives_single_tile():
    image, sheet = make_sheet()
    ((tile_image, tile),) = tile_sheet(image, sheet, (500, 500))
    assert tile_image.size == (W, H)
    assert [b.bbox for b in tile.boxes] == [b.bbox for b in sheet.boxes]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": (0, 30)},
        {"tile_size": (50, 30), "overlap": 30},
        {"tile_size": (50, 30), "overlap": -1},
        {"tile_size": (50, 30), "legend_mode": "bogus"},
    ],
)
def test_bad_parameters_rejected(kwargs):
    image, sheet = make_sheet()
    with pytest.raises(ValueError):
        tile_sheet(image, sheet, **kwargs)


def test_with_legend_pastes_missing_legend_under_tile():
    image, sheet = make_sheet()
    tiles = by_name(tile_sheet(image, sheet, (50, 30), legend_mode="with-legend"))
    # правая нижняя плитка: был только объект (60,40), образец лежит в левой верхней
    tile_image, tile = tiles["plan__tile_x50_y30.jpg"]
    pasted = [b for b in tile.boxes if b.category == LEGEND_ITEM]
    assert len(pasted) == 1 and pasted[0].attributes["pasted"] is True
    assert tile.height > 30 and tile_image.size == (tile.width, tile.height)
    assert pasted[0].bbox[1] >= 30  # образец в полосе под плиткой
    assert all(b.attributes["legend_in_tile"] for b in tile.boxes if b.category == OBJECT)
    assert_boxes_on_marks(tile_image, tile)
    # плитка, где образец и так есть, не меняется
    own_image, own = tiles["plan__tile_x0_y0.jpg"]
    assert own.height == 30 and not any(b.attributes.get("pasted") for b in own.boxes)


def test_with_legend_skips_legend_wider_than_tile():
    image, sheet = make_sheet()
    # образец 10 px + отступы 2*8 не влезает в плитку шириной 12: приклеить нечего
    for tile_image, tile in tile_sheet(image, sheet, (12, 30), legend_mode="with-legend"):
        assert not any(b.attributes.get("pasted") for b in tile.boxes)
        assert tile.height == 30
