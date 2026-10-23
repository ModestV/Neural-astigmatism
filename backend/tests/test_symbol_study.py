"""Проверки протокола исследования: эталоны, отсутствие утечки и кэш признаков."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "experiments/vozderjus/symbol_matching"
sys.path.insert(0, str(ROOT))

from annotation_core import AnnotationPage, Mark  # noqa: E402
from evaluation import evaluate  # noqa: E402
from models import Box, Settings  # noqa: E402
from study import page_target, select_threshold  # noqa: E402


def sample_page():
    return AnnotationPage(
        id="p",
        file="a.pdf",
        page=1,
        width=200,
        height=200,
        sha256="x",
        annotations=[
            Mark(id="q", symbol="lamp", legend=True, bbox=Box(x=10, y=10, w=10, h=10)),
            Mark(id="t", symbol="lamp", bbox=Box(x=100, y=100, w=10, h=10)),
        ],
    )


def test_legend_does_not_inflate_recall():
    target = page_target(sample_page())
    result = evaluate([], target, 0.5, False)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)


def test_threshold_selection_ignores_holdout():
    target = page_target(sample_page())
    prediction = {
        "symbol": "lamp",
        "template_id": "q",
        "bbox": {"x": 100, "y": 100, "w": 10, "h": 10},
        "score": 0.82,
        "angle": 0,
        "scale": 1,
    }
    fp = {**prediction, "bbox": {"x": 150, "y": 150, "w": 10, "h": 10}, "score": 0.78}
    records = [
        {
            "file": "a.pdf",
            "algorithm": "template",
            "split": "calibration",
            "detections": [prediction, fp],
        },
        {
            "file": "a.pdf",
            "algorithm": "template",
            "split": "test",
            "detections": [{**fp, "score": 0.94} for _ in range(30)],
        },
    ]
    threshold, _ = select_threshold(records, {"a.pdf": target}, "template")
    assert threshold == 0.8


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_shared_target_features_preserve_detections(method):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from algorithms.features import detect, prepare_target

    cv2.setNumThreads(1)
    rng = np.random.default_rng(33)
    patch = np.full((120, 120), 255, dtype=np.uint8)
    for x, y in rng.integers(12, 108, (80, 2)):
        cv2.circle(patch, (int(x), int(y)), 3, 0, -1)
    image = np.full((300, 400), 255, dtype=np.uint8)
    image[50:170, 40:160] = patch
    settings = Settings(angles=[0], scales=[1], max_detections=10)
    direct, _ = detect(image, patch, "q", "lamp", settings, method)
    shared = prepare_target(image, settings, method)
    cached, _ = detect(image, patch, "q", "lamp", settings, method, target_features=shared)
    assert direct
    assert [d.model_dump() for d in direct] == [d.model_dump() for d in cached]


@pytest.mark.parametrize("shape", [(35, 45), (280, 310)])
def test_tiled_peaks_equal_full_scan_with_ties(shape):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from algorithms.template_matching import peak_locations

    response = np.random.default_rng(71).random(shape, dtype=np.float32)
    response[5:15, 7:18] = 1.0
    legacy = response.copy()
    expected = []
    for _ in range(100):
        _, score, _, (x, y) = cv2.minMaxLoc(legacy)
        if score < 0.8:
            break
        expected.append((score, x, y))
        legacy[max(0, y - 9) : y + 10, max(0, x - 12) : x + 13] = -1
    actual, _ = peak_locations(response.copy(), 0.8, 24, 18, 100)
    assert actual == expected
