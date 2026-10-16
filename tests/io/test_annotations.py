from __future__ import annotations

import json
from pathlib import Path

import pytest

from symspot.io.annotations import load_ground_truth, load_queries, load_similar_groups


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _ann(aid, image_id, cat, bbox, **attrs):
    return {
        "id": aid,
        "image_id": image_id,
        "category_id": cat,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "attributes": {"occluded": False, **attrs},
    }


def _coco_single(tmp_path: Path) -> Path:
    data = {
        "images": [{"id": 1, "file_name": "images/plan.png", "width": 1000, "height": 500}],
        "categories": [{"id": 1, "name": "legend_item"}, {"id": 2, "name": "object"}],
        "annotations": [
            _ann(1, 1, 1, [10, 20, 30, 40], legend_id=3, meaning="duplex"),
            _ann(2, 1, 1, [100, 20, 30, 40], legend_id=" 4 ", meaning=""),
            _ann(3, 1, 1, [200, 20, 30, 40], legend_id="3"),  # второй эталон класса 3
            _ann(4, 1, 2, [300, 100, 20, 10], legend_id="3"),
            _ann(5, 1, 2, [400, 100, 20, 10], legend_id=4.0),
            _ann(6, 1, 3, [0, 0, 5, 5], legend_id="9"),  # чужая категория
        ],
    }
    data["categories"].append({"id": 3, "name": "wall"})
    return _write(tmp_path / "instances_Train.json", data)


def _coco_multi(tmp_path: Path) -> Path:
    data = {
        "images": [
            {"id": 1, "file_name": "a/plan_A.png", "width": 100, "height": 100},
            {"id": 2, "file_name": "b/plan_B.png", "width": 200, "height": 100},
        ],
        "categories": [{"id": 1, "name": "legend_item"}, {"id": 2, "name": "object"}],
        "annotations": [
            _ann(1, 1, 1, [0, 0, 10, 10], legend_id="1"),
            _ann(2, 2, 1, [20, 0, 10, 10], legend_id="2"),
            _ann(3, 2, 2, [40, 0, 10, 10], legend_id="2"),
        ],
    }
    return _write(tmp_path / "multi.json", data)


def test_coco_cvat_queries(tmp_path):
    qs = load_queries(_coco_single(tmp_path))
    assert [q.legend_id for q in qs] == ["3", "4", "3"]  # число -> str, пробелы обрезаны
    assert qs[0].bbox == (10.0, 20.0, 40.0, 60.0)  # xywh -> xyxy
    assert qs[0].label == "duplex"
    assert qs[1].label is None  # пустой meaning
    assert qs[2].label is None


def test_coco_cvat_ground_truth(tmp_path):
    gt = load_ground_truth(_coco_single(tmp_path))
    assert [g.legend_id for g in gt] == ["3", "4"]  # 4.0 -> "4"; категория wall пропущена
    assert gt[0].bbox == (300.0, 100.0, 320.0, 110.0)
    assert all(not g.is_query and g.parent_ids == [] for g in gt)


def test_coco_scaling_with_drawing_size(tmp_path):
    path = _coco_single(tmp_path)
    qs = load_queries(path, drawing_size=(500.0, 1000.0))  # sx = 0.5, sy = 2
    assert qs[0].bbox == pytest.approx((5.0, 40.0, 20.0, 120.0))
    gt = load_ground_truth(path, drawing_size=(500.0, 1000.0))
    assert gt[0].bbox == pytest.approx((150.0, 200.0, 160.0, 220.0))
    # без drawing_size масштаб не применяется
    assert load_queries(path)[0].bbox == (10.0, 20.0, 40.0, 60.0)


def test_coco_image_selection_by_name(tmp_path):
    path = _coco_multi(tmp_path)
    qs = load_queries(path, image_name="/some/dir/plan_B.pdf")
    assert [(q.legend_id, q.bbox) for q in qs] == [("2", (20.0, 0.0, 30.0, 10.0))]
    gt = load_ground_truth(path, image_name="plan_A.png")
    assert gt == []  # у первого изображения нет object
    gt = load_ground_truth(path, drawing_size=(400.0, 100.0), image_name="plan_B")
    assert gt[0].bbox == pytest.approx((80.0, 0.0, 100.0, 10.0))


@pytest.mark.parametrize("name", [None, "other.png"])
def test_coco_ambiguous_image_raises(tmp_path, name):
    with pytest.raises(ValueError, match="plan_A"):
        load_queries(_coco_multi(tmp_path), image_name=name)


def test_empty_legend_id_raises_with_annotation_id(tmp_path):
    data = {
        "images": [{"id": 1, "file_name": "p.png", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "legend_item"}],
        "annotations": [_ann(77, 1, 1, [0, 0, 1, 1], legend_id="   ")],
    }
    with pytest.raises(ValueError, match="77"):
        load_queries(_write(tmp_path / "bad.json", data))
    data["annotations"] = [_ann(78, 1, 1, [0, 0, 1, 1])]  # атрибута нет совсем
    with pytest.raises(ValueError, match="78"):
        load_queries(_write(tmp_path / "bad2.json", data))


def test_simple_json_list(tmp_path):
    path = _write(
        tmp_path / "q.json",
        [
            {"legend_id": 1, "bbox": [10, 10, 20, 30], "label": "a"},
            {"legend_id": "2", "bbox": [30, 30, 5, 5]},  # перепутанный порядок углов
        ],
    )
    qs = load_queries(path)
    assert [q.legend_id for q in qs] == ["1", "2"]
    assert qs[1].bbox == (5.0, 5.0, 30.0, 30.0)
    assert qs[0].label == "a" and qs[1].label is None
    with pytest.raises(ValueError):
        load_ground_truth(path)


def test_simple_json_dict_with_scaling(tmp_path):
    path = _write(
        tmp_path / "q.json",
        {
            "image": "plan.png",
            "width": 100,
            "height": 200,
            "legend_items": [{"legend_id": "x", "bbox": [10, 20, 30, 40]}],
        },
    )
    (q,) = load_queries(path, drawing_size=(200.0, 100.0))
    assert q.bbox == pytest.approx((20.0, 10.0, 60.0, 20.0))
    (q,) = load_queries(path)
    assert q.bbox == (10.0, 20.0, 30.0, 40.0)


def _synth(tmp_path: Path) -> Path:
    data = {
        "name": "s0",
        "width": 100.0,
        "height": 50.0,
        "units": "pt",
        "similar_groups": [["duplex", "gfci", "r220"], ["a", "b"]],
        "queries": [{"legend_id": "duplex", "bbox": [1, 2, 11, 12], "label": "Duplex"}],
        "instances": [
            {
                "legend_id": "duplex",
                "T": {"s": 1.0, "theta": 0.0, "tx": 0.0, "ty": 0.0, "mirror": False},
                "bbox": [1, 2, 11, 12],
                "path_ids": ["i0_0"],
                "is_query": True,
                "hard": [],
            },
            {
                "legend_id": "gfci",
                "T": {"s": 1.0, "theta": 0.0, "tx": 5.0, "ty": 5.0, "mirror": False},
                "bbox": [20, 10, 30, 20],
                "path_ids": ["i1_0", "i1_1"],
                "is_query": False,
                "hard": ["rot"],
            },
        ],
    }
    return _write(tmp_path / "gt.json", data)


def test_synth_ground_truth(tmp_path):
    path = _synth(tmp_path)
    gt = load_ground_truth(path)
    assert [(g.legend_id, g.is_query) for g in gt] == [("duplex", True), ("gfci", False)]
    assert gt[1].bbox == (20.0, 10.0, 30.0, 20.0)
    assert gt[1].parent_ids == []
    (q,) = load_queries(path)
    assert (q.legend_id, q.bbox, q.label) == ("duplex", (1.0, 2.0, 11.0, 12.0), "Duplex")
    assert load_similar_groups(path) == [{"duplex", "gfci", "r220"}, {"a", "b"}]


def test_synth_scaling(tmp_path):
    gt = load_ground_truth(_synth(tmp_path), drawing_size=(200.0, 100.0))
    assert gt[1].bbox == pytest.approx((40.0, 20.0, 60.0, 40.0))


def test_similar_groups_empty_for_coco_and_simple(tmp_path):
    assert load_similar_groups(_coco_single(tmp_path)) == []
    assert load_similar_groups(_write(tmp_path / "s.json", [])) == []


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        load_queries(_write(tmp_path / "x.json", {"foo": 1}))
