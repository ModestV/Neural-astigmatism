"""Проверки исследования: поворот PDF, метрики и несколько экземпляров."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = (Path(__file__).resolve().parents[2]
        / "experiments/vozderjus/symbol_matching")
sys.path.insert(0, str(ROOT))

from algorithms.common import iou  # noqa: E402
from evaluation import evaluate  # noqa: E402
from models import Box, Detection, Experiment, Settings, TargetSpec  # noqa: E402
from pdf_io import crop_template, pdf_path, render_page  # noqa: E402


def prediction(x=10, score=0.9, symbol="lamp"):
    return Detection(
        symbol=symbol,
        template_id="q",
        bbox=Box(x=x, y=10, w=20, h=20),
        score=score,
        angle=0,
        scale=1,
    )


def test_metrics_no_labels_and_duplicate_matching():
    target = TargetSpec(file="a.pdf")
    assert evaluate([prediction()], target, 0.5, False) is None
    target = TargetSpec(
        file="a.pdf",
        ground_truth=[{"symbol": "lamp", "bbox": {"x": 10, "y": 10, "w": 20, "h": 20}}],
    )
    result = evaluate([prediction(), prediction(), prediction(x=100)], target, 0.5, False)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 1
    assert result["f1"] == 0.5


def test_legends_and_classes():
    target = TargetSpec(
        file="a.pdf",
        ground_truth=[
            {
                "symbol": "lamp",
                "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
                "legend": True,
            },
            {"symbol": "lamp", "bbox": {"x": 100, "y": 10, "w": 20, "h": 20}},
        ],
    )
    result = evaluate([prediction(), prediction(x=100, symbol="socket")], target, 0.5, False)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    result = evaluate([prediction()], target, 0.5, True)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)


def test_known_empty_is_not_unlabeled():
    target = TargetSpec(file="a.pdf", ground_truth=[])
    result = evaluate([prediction()], target, 0.5, False)
    assert result["fp"] == 1
    assert result["f1"] == 0
    assert result["recall"] is None


def test_invalid_config_and_paths(tmp_path):
    with pytest.raises(ValueError):
        Settings(scales=[0])
    with pytest.raises(ValueError):
        Settings(angles=[float("nan")])
    with pytest.raises(ValueError):
        pdf_path(tmp_path, "../secret.pdf")
    with pytest.raises(ValueError):
        Experiment(templates=[], targets=[])


def test_rotated_pdf_render_and_cache(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("cv2")
    path = tmp_path / "поворот.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=100)
        page.draw_rect(pymupdf.Rect(10, 10, 40, 30), color=(0, 0, 0), fill=(0, 0, 0))
        page.set_rotation(270)
        doc.save(path)
    image, meta = render_page(path, 1, 144, tmp_path / "cache", 1_000_000)
    assert image.shape == (400, 200)
    assert meta["rotation"] == 270
    # При 270° исходный прямоугольник расположен внизу слева.
    patch = crop_template(image, Box(x=10, y=160, w=20, h=30), meta)
    assert patch.mean() < 20
    _, again = render_page(path, 1, 144, tmp_path / "cache", 1_000_000)
    assert again["cache_hit"]
    with pytest.raises(ValueError):
        render_page(path, 1, 144, tmp_path / "cache", 100)
    with pytest.raises(ValueError):
        render_page(path, 2, 144, tmp_path / "cache", 1_000_000)


@pytest.mark.parametrize("algorithm", ["template", "orb", "sift"])
def test_two_instances_with_rotation(algorithm):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from algorithms.features import detect as features
    from algorithms.template_matching import detect as template_matching

    rng = np.random.default_rng(124)
    template = np.full((120, 120), 255, dtype=np.uint8)
    for _ in range(65):
        x, y = rng.integers(12, 108, 2)
        cv2.circle(template, (int(x), int(y)), int(rng.integers(2, 6)), 0, -1)
    image = np.full((400, 600), 255, dtype=np.uint8)
    image[40:160, 50:170] = template
    image[220:340, 400:520] = cv2.rotate(template, cv2.ROTATE_180)
    settings = Settings(
        angles=[0, 180],
        scales=[1],
        max_detections=20,
        template_threshold=0.95,
        features=5000,
    )
    if algorithm == "template":
        found, info = template_matching(image, template, "q", "symbol", settings)
    else:
        found, info = features(image, template, "q", "symbol", settings, algorithm)
    for box in (Box(x=50, y=40, w=120, h=120), Box(x=400, y=220, w=120, h=120)):
        assert any(iou(d.bbox, box) > 0.85 for d in found), info


def test_template_without_features_has_explicit_status():
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from algorithms.features import detect

    empty = np.full((40, 40), 255, dtype=np.uint8)
    found, info = detect(empty, empty, "q", "symbol", Settings(), "orb")
    assert not found
    assert info["status"] == "insufficient_keypoints"


def test_cli_from_other_directory_without_truth(tmp_path):
    import json
    import subprocess

    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("cv2")
    inputs = tmp_path / "чертежи"
    inputs.mkdir()
    with pymupdf.open() as document:
        page = document.new_page(width=120, height=100)
        page.draw_circle((20, 20), 7, color=(0, 0, 0))
        document.save(inputs / "лист.pdf")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "id": "q",
                        "symbol": "circle",
                        "file": "лист.pdf",
                        "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
                    }
                ],
                "targets": [{"file": "лист.pdf"}],
            }
        ),
        encoding="utf-8",
    )
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_benchmark.py"),
            "run",
            "--config",
            str(config),
            "--drawings-dir",
            str(inputs),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "results"),
            "--algorithms",
            "template",
            "--dpi",
            "72",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    folder = next((tmp_path / "results").iterdir())
    record = json.loads((folder / "001_template.json").read_text())
    assert record["metrics"] is None
    assert len(record["detections"]) == 0  # Сам образец исключён из результата.
    assert json.loads((folder / "status.json").read_text())["status"] == "complete"
