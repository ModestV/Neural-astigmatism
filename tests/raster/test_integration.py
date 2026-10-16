"""Растровый конвейер целиком на синтетике (PLAN, фаза 5: F1 чистый >= 0.85, грязный >= 0.7)."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from symspot.core.config import Config
from symspot.eval.metrics import evaluate
from symspot.io.annotations import load_ground_truth
from symspot.pipeline import run
from symspot.synth.export import generate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("raster_synth")
    generate(1, "small", 0, out, raster=True)
    return out


def _f1(directory: Path, suffix: str) -> float:
    png = directory / f"small_000{suffix}.png"
    coco = directory / f"small_000{suffix}_coco.json"
    h, w = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE).shape
    result, _ = run(png, coco, Config())
    assert result.kind == "raster"
    gt = load_ground_truth(coco, drawing_size=(w, h), image_name=str(png))
    return evaluate(result.instances, gt).overall.f1


def test_clean_png_f1(synth_dir: Path) -> None:
    assert _f1(synth_dir, "") >= 0.85


def test_dirty_png_f1(synth_dir: Path) -> None:
    assert _f1(synth_dir, "_dirty") >= 0.7
