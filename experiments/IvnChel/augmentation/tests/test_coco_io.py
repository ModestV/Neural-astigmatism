"""Тесты чтения и записи COCO: на маленьком примере и на реальной разметке из git."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coco_io import LEGEND_ITEM, OBJECT, load_sheets, save_coco  # noqa: E402

ANNOTATIONS = Path(__file__).resolve().parents[4] / "datasets" / "labeled-dataset" / "annotations"


def make_coco(path: Path, object_id: int, legend_id: int) -> None:
    """Маленький COCO-файл с двумя листами: один размечен, второй пустой."""
    data = {
        "categories": [
            {"id": legend_id, "name": "legend_item"},
            {"id": object_id, "name": "object"},
            {"id": 9, "name": "temporary"},
        ],
        "images": [
            {"id": 1, "width": 100, "height": 50, "file_name": "a.jpg"},
            {"id": 2, "width": 100, "height": 50, "file_name": "empty.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": legend_id, "bbox": [1, 2, 10, 5],
             "attributes": {"legend_id": "7", "rotation": 0.0}},
            {"id": 2, "image_id": 1, "category_id": object_id, "bbox": [30, 20, 10, 5],
             "attributes": {"legend_id": "7", "rotation": 90.0}},
            {"id": 3, "image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1], "attributes": {}},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_classes_matched_by_name_not_by_id(tmp_path):
    make_coco(tmp_path / "p1.json", object_id=1, legend_id=2)
    make_coco(tmp_path / "p2.json", object_id=2, legend_id=1)
    sheets = load_sheets([tmp_path / "p1.json", tmp_path / "p2.json"])
    assert len(sheets) == 2  # пустые листы пропущены
    for sheet in sheets:
        assert [b.category for b in sheet.boxes] == [LEGEND_ITEM, OBJECT]  # temporary выброшен
        assert sheet.boxes[0].legend_id == sheet.boxes[1].legend_id == "7"


def test_keep_empty_sheets_on_request(tmp_path):
    make_coco(tmp_path / "p.json", object_id=1, legend_id=2)
    assert len(load_sheets([tmp_path / "p.json"], skip_empty=False)) == 2


def test_save_and_load_roundtrip(tmp_path):
    make_coco(tmp_path / "p.json", object_id=1, legend_id=2)
    sheets = load_sheets([tmp_path / "p.json"])
    save_coco(sheets, tmp_path / "out.json")
    again = load_sheets([tmp_path / "out.json"])
    assert [(s.file_name, s.width, s.height) for s in again] == [("a.jpg", 100, 50)]
    assert [(b.category, b.bbox, b.attributes) for b in again[0].boxes] == [
        (b.category, b.bbox, b.attributes) for b in sheets[0].boxes
    ]
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert {c["name"]: c["id"] for c in out["categories"]} == {"legend_item": 1, "object": 2}


def test_real_annotations_match_readme():
    sheets = load_sheets(sorted(ANNOTATIONS.glob("*.json")))
    boxes = [b for s in sheets for b in s.boxes]
    assert sum(b.category == OBJECT for b in boxes) == 719 + 1189
    assert sum(b.category == LEGEND_ITEM for b in boxes) == 135 + 253
    assert not any(s.file_name.startswith("03_washoe_house") for s in sheets)


def test_find_problems_detects_broken_annotations(tmp_path):
    from PIL import Image

    from coco_io import Box, Sheet, find_problems

    Image.new("L", (100, 50), 255).save(tmp_path / "ok.png")
    good = Sheet("ok.png", 100, 50, [
        Box(LEGEND_ITEM, (0, 0, 10, 10), {"legend_id": "1"}),
        Box(OBJECT, (90, 40, 10, 10), {"legend_id": "1", "legend_in_tile": True}),
    ])
    assert find_problems([good], tmp_path) == []

    outside = Sheet("ok.png", 100, 50, [Box(OBJECT, (95, 40, 10, 10), {"legend_id": "1"})])
    assert any("выходит за картинку" in p for p in find_problems([outside]))
    empty = Sheet("ok.png", 100, 50, [Box(OBJECT, (1, 1, 0, 5), {})])
    assert any("размер рамки" in p for p in find_problems([empty]))
    stale = Sheet("ok.png", 100, 50, [Box(OBJECT, (1, 1, 5, 5), {"legend_id": "9",
                                                                   "legend_in_tile": True})])
    assert any("legend_in_tile" in p for p in find_problems([stale]))
    missing = Sheet("nope.png", 100, 50, [])
    assert any("файла нет" in p for p in find_problems([missing], tmp_path))
    wrong_size = Sheet("ok.png", 120, 50, [])
    assert any("размер файла" in p for p in find_problems([wrong_size], tmp_path))
    assert any("больше одного раза" in p for p in find_problems([good, good]))


def test_collect_annotation_paths_expands_folders(tmp_path):
    from coco_io import collect_annotation_paths

    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "x.txt").write_text("")
    single = tmp_path / "x.txt"
    assert collect_annotation_paths([tmp_path, single]) == [
        tmp_path / "a.json", tmp_path / "b.json", single
    ]
