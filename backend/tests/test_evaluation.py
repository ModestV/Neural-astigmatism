from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from app.detection import get_detector
from app.evaluation import Annotations, Instance, evaluate, iou, load, save
from app.evaluation.dataset import DatasetFormatError, from_dict, load_predictions
from app.evaluation.metrics import average_precision, match, pr_curve
from app.evaluation.report import write_all
from app.evaluation.run import run_detector

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"


def gt(*items: tuple, legend: bool = False) -> list[Instance]:
    return [Instance("a.pdf", 1, s, b, legend=legend) for s, b in items]


def pred(symbol: str, bbox: tuple, score: float = 1.0, page: int = 1) -> Instance:
    return Instance("a.pdf", page, symbol, bbox, score=score)


def ann(instances: list[Instance], **kwargs) -> Annotations:
    return Annotations(instances=instances, **kwargs)


# --- IoU ------------------------------------------------------------------------


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0
    assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0  # касаются стороной
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


# --- сопоставление и метрики ------------------------------------------------------


def test_perfect_predictions():
    boxes = [("socket", (0, 0, 10, 10)), ("lamp", (50, 50, 10, 10))]
    report = evaluate(ann(gt(*boxes)), ann([pred(s, b, 0.9) for s, b in boxes]))
    o = report.overall
    assert (o.tp, o.fp, o.fn) == (2, 0, 0)
    assert o.precision == o.recall == o.f1 == o.mean_iou == o.ap == 1
    assert report.macro["map"] == 1


def test_missed_and_false_positive():
    truth = ann(gt(("socket", (0, 0, 10, 10)), ("socket", (50, 0, 10, 10))))
    predictions = ann([pred("socket", (0, 0, 10, 10)), pred("socket", (100, 100, 10, 10))])
    o = evaluate(truth, predictions).overall
    assert (o.tp, o.fp, o.fn) == (1, 1, 1)
    assert o.precision == o.recall == o.f1 == 0.5


def test_duplicate_prediction_is_false_positive():
    truth = ann(gt(("socket", (0, 0, 10, 10))))
    predictions = ann([pred("socket", (0, 0, 10, 10), 0.9), pred("socket", (1, 0, 10, 10), 0.8)])
    m = match(truth, predictions)
    assert [p.truth is not None for p in m.predictions] == [True, False]
    o = evaluate(truth, predictions).overall
    assert (o.tp, o.fp, o.fn) == (1, 1, 0)


def test_higher_score_wins_the_match():
    truth = ann(gt(("socket", (0, 0, 10, 10))))
    good, bad = pred("socket", (0, 0, 10, 10), 0.6), pred("socket", (2, 0, 10, 10), 0.9)
    m = match(truth, ann([good, bad]))
    assert m.predictions[0].prediction == bad and m.predictions[0].truth is not None
    assert m.predictions[1].truth is None


def test_iou_threshold():
    truth = ann(gt(("socket", (0, 0, 10, 10))))
    predictions = ann([pred("socket", (5, 0, 10, 10))])  # IoU = 1/3
    assert evaluate(truth, predictions, iou_threshold=0.5).overall.tp == 0
    assert evaluate(truth, predictions, iou_threshold=0.3).overall.tp == 1


def test_wrong_symbol_is_miss_and_false_positive():
    truth = ann(gt(("socket", (0, 0, 10, 10))))
    report = evaluate(truth, ann([pred("lamp", (0, 0, 10, 10))]))
    assert (report.overall.tp, report.overall.fp, report.overall.fn) == (0, 1, 1)
    by_symbol = {s.symbol: s for s in report.per_symbol}
    assert by_symbol["socket"].fn == 1 and by_symbol["lamp"].fp == 1
    assert by_symbol["lamp"].recall is None


def test_other_page_does_not_match():
    truth = ann(gt(("socket", (0, 0, 10, 10))))
    report = evaluate(truth, ann([pred("socket", (0, 0, 10, 10), page=2)]))
    assert (report.overall.tp, report.overall.fp, report.overall.fn) == (0, 1, 1)


def test_legend_is_ignored_by_default():
    truth = ann(gt(("socket", (0, 0, 10, 10)), legend=True) + gt(("socket", (50, 0, 10, 10))))
    predictions = ann([pred("socket", (0, 0, 10, 10)), pred("socket", (50, 0, 10, 10))])
    report = evaluate(truth, predictions)
    assert (report.overall.tp, report.overall.fp, report.overall.fn) == (1, 0, 0)
    assert report.ignored_predictions == 1
    with_legend = evaluate(truth, predictions, include_legend=True)
    assert (with_legend.overall.tp, with_legend.overall.n_truth) == (2, 2)


def test_score_threshold_turns_hits_into_misses():
    truth = ann(gt(("socket", (0, 0, 10, 10)), ("socket", (50, 0, 10, 10))))
    predictions = ann([pred("socket", (0, 0, 10, 10), 0.9), pred("socket", (50, 0, 10, 10), 0.3)])
    o = evaluate(truth, predictions, score_threshold=0.5).overall
    assert (o.tp, o.fp, o.fn) == (1, 0, 1)
    assert o.ap == 1  # AP считается по всем ответам, порог на неё не влияет


def test_no_predictions():
    report = evaluate(ann(gt(("socket", (0, 0, 10, 10)))), ann([]))
    o = report.overall
    assert o.precision is None and o.recall == 0 and o.f1 == 0 and o.ap == 0
    assert report.pr_curve == []


# --- PR-кривая и AP -------------------------------------------------------------


def test_pr_curve_and_average_precision():
    # Эталонов 2; ответы по убыванию уверенности: TP, FP, TP.
    truth = ann(gt(("socket", (0, 0, 10, 10)), ("socket", (50, 0, 10, 10))))
    predictions = ann(
        [
            pred("socket", (0, 0, 10, 10), 0.9),
            pred("socket", (100, 0, 10, 10), 0.8),
            pred("socket", (50, 0, 10, 10), 0.7),
        ]
    )
    m = match(truth, predictions)
    points = pr_curve(m.predictions, m.n_truth)
    assert [(p.threshold, p.precision, p.recall) for p in points] == [
        (0.9, 1.0, 0.5),
        (0.8, 0.5, 0.5),
        (0.7, pytest.approx(2 / 3), 1.0),
    ]
    # Огибающая: до recall 0.5 точность 1, дальше 2/3 → 0.5·1 + 0.5·2/3
    assert average_precision(points) == pytest.approx(0.5 + 1 / 3)
    assert evaluate(truth, predictions).overall.ap == pytest.approx(0.5 + 1 / 3)


def test_pr_curve_groups_equal_scores():
    truth = ann(gt(("socket", (0, 0, 10, 10)), ("socket", (50, 0, 10, 10))))
    predictions = ann([pred("socket", (0, 0, 10, 10), 0.5), pred("socket", (50, 0, 10, 10), 0.5)])
    m = match(truth, predictions)
    assert len(pr_curve(m.predictions, m.n_truth)) == 1


# --- время ----------------------------------------------------------------------


def test_timing():
    truth = ann(
        [Instance("a.pdf", 1, "s", (0, 0, 1, 1)), Instance("a.pdf", 2, "s", (0, 0, 1, 1))]
        + [Instance("b.pdf", 1, "s", (0, 0, 1, 1))]
    )
    t = evaluate(truth, ann([], times={"a.pdf": 3.0, "b.pdf": 1.0})).timing
    assert (t.files, t.pages, t.total_s) == (2, 3, 4.0)
    assert (t.mean_per_file_s, t.max_per_file_s, t.mean_per_page_s) == (
        2.0,
        3.0,
        pytest.approx(4 / 3),
    )
    assert evaluate(truth, ann([])).timing is None


# --- форматы файлов -------------------------------------------------------------


def test_load_ground_truth_sample():
    truth = load(SAMPLES / "ground_truth.json")
    assert set(truth.files) == {"plan.svg", "plan-2pages.pdf", "plan.dxf"}
    assert truth.symbol_names["socket"] == "Розетка"
    assert sum(i.legend for i in truth.instances) == 16


def test_predictions_roundtrip(tmp_path):
    original = ann([pred("socket", (1.5, 2, 3, 4), 0.75)], times={"a.pdf": 0.5}, detector="my")
    save(original, tmp_path / "p.json")
    loaded = load(tmp_path / "p.json")
    assert loaded.instances == original.instances
    assert (loaded.times, loaded.detector) == (original.times, "my")


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"files": {"a.pdf": {"pages": {"1": [{"symbol": "s", "bbox": [0, 0, 0, 5]}]}}}},
        {"files": {"a.pdf": {"pages": {"1": [{"bbox": [0, 0, 5, 5]}]}}}},
        {"files": {"a.pdf": {"pages": {"x": [{"symbol": "s", "bbox": [0, 0, 5, 5]}]}}}},
    ],
)
def test_bad_format(data):
    with pytest.raises(DatasetFormatError):
        from_dict(data)


# --- COCO ------------------------------------------------------------------------


def coco(file_name: str = "plan.jpg", image_id: int = 1, annotations: list | None = None) -> dict:
    return {
        "images": [{"id": image_id, "file_name": file_name, "width": 100, "height": 100}],
        "categories": [
            {"id": 1, "name": "switch"},
            {"id": 2, "name": "light"},
            {"id": 3, "name": "legend_item"},
        ],
        "annotations": annotations
        if annotations is not None
        else [
            {
                "id": 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [10, 10, 5, 8],
                "attributes": {"occluded": False, "rotation": 90.0},
            },
            {"id": 2, "image_id": image_id, "category_id": 2, "bbox": [50, 50, 6, 6]},
            {"id": 3, "image_id": image_id, "category_id": 3, "bbox": [80, 5, 10, 10]},
        ],
    }


def write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_load_coco(tmp_path):
    truth = load(write_json(tmp_path / "gt.json", coco()))
    assert truth.files == ["plan.jpg"]
    switch, light, legend = truth.instances
    assert (switch.symbol, switch.page, switch.bbox, switch.angle) == (
        "switch",
        1,
        (10, 10, 5, 8),
        90,
    )
    assert light.symbol == "light" and not light.legend
    assert legend.legend


def test_load_several_coco_files(tmp_path):
    a = write_json(tmp_path / "a.json", coco("a.jpg"))
    b = write_json(tmp_path / "b.json", coco("b.jpg"))
    truth = load(a, b)
    assert truth.files == ["a.jpg", "b.jpg"]
    assert len(truth.instances) == 6


def test_coco_results(tmp_path):
    gt_path = write_json(tmp_path / "gt.json", coco())
    results = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 5, 8], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 6, 6], "score": 0.4},
    ]
    predictions = load_predictions(write_json(tmp_path / "res.json", results), [gt_path])
    assert [(p.file, p.symbol, p.score) for p in predictions.instances] == [
        ("plan.jpg", "switch", 0.9),
        ("plan.jpg", "light", 0.4),
    ]
    o = evaluate(load(gt_path), predictions).overall
    assert (o.tp, o.fp, o.fn) == (1, 1, 1)  # legend_item не оценивается


def test_coco_results_ambiguous_image_id(tmp_path):
    gts = [write_json(tmp_path / f"{n}.json", coco(f"{n}.jpg")) for n in "ab"]
    res = write_json(
        tmp_path / "res.json", [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}]
    )
    with pytest.raises(DatasetFormatError, match="неоднозначна"):
        load_predictions(res, gts)
    named = write_json(
        tmp_path / "named.json",
        [{"image_id": 1, "file_name": "b.jpg", "category_id": 1, "bbox": [0, 0, 1, 1]}],
    )
    assert load_predictions(named, gts).instances[0].file == "b.jpg"


def test_coco_results_need_coco_ground_truth(tmp_path):
    res = write_json(
        tmp_path / "res.json", [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}]
    )
    with pytest.raises(DatasetFormatError, match="COCO"):
        load_predictions(res, [SAMPLES / "ground_truth.json"])


def test_bad_coco(tmp_path):
    data = coco(annotations=[{"id": 1, "image_id": 7, "category_id": 1, "bbox": [0, 0, 1, 1]}])
    with pytest.raises(DatasetFormatError):
        load(write_json(tmp_path / "gt.json", data))


# --- прогон алгоритма и командная строка --------------------------------------


def test_run_stub_detector_on_samples(tmp_path):
    truth = load(SAMPLES / "ground_truth.json")
    predictions = run_detector(get_detector("stub"), truth, SAMPLES)
    assert set(predictions.times) == set(truth.files)
    # Заглушка находит только сами образцы: первую легенду каждого чертежа (4 обозначения × 3 файла)
    report = evaluate(truth, predictions, include_legend=True)
    assert (report.overall.tp, report.overall.fp) == (12, 0)
    assert evaluate(truth, predictions).overall.tp == 0
    write_all(report, tmp_path, truth.symbol_names)
    assert {p.name for p in tmp_path.iterdir()} == {
        "metrics.json",
        "summary.md",
        "pr_curve.csv",
        "pr_curve.svg",
    }
    assert json.loads((tmp_path / "metrics.json").read_text())["overall"]["tp"] == 12


def test_cli_simulate(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--simulate",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "macro F1" in result.stdout
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["detector"] == "simulated"
    assert 0 < metrics["overall"]["recall"] < 1
    assert (tmp_path / "predictions.json").is_file()


def test_cli_coco_with_coco_results(tmp_path):
    gt_path = write_json(tmp_path / "gt.json", coco())
    res = write_json(
        tmp_path / "res.json",
        [{"image_id": 1, "category_id": 2, "bbox": [50, 50, 6, 6], "score": 0.8}],
    )
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--gt",
            str(gt_path),
            "--pred",
            str(res),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert (metrics["overall"]["tp"], metrics["overall"]["fn"]) == (1, 1)
