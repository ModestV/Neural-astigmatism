"""Бенчмарк фазы 6: large вектор < 60 с, растр 7200x4800 со всеми эталонами < 3 мин.

Запуск: uv run pytest tests/bench -m bench -q -s
"""

from __future__ import annotations

import time

import pytest

from symspot.core.config import Config
from symspot.eval.metrics import evaluate
from symspot.io.annotations import load_ground_truth
from symspot.pipeline import run
from symspot.synth.export import generate

pytestmark = pytest.mark.bench

VECTOR_LIMIT_S = 60.0
RASTER_LIMIT_S = 180.0


@pytest.fixture(scope="module")
def large(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    (name,) = generate(1, "large", 0, out, raster=True)
    return out, name


def _run(path, gt, drawing_kind_gt):
    t0 = time.perf_counter()
    res, d = run(path, gt, Config())
    dt = time.perf_counter() - t0
    g = load_ground_truth(gt, drawing_size=(d.width, d.height), image_name=str(path))
    m = evaluate(res.instances, g)
    print(f"\n{path.name}: {dt:.1f} s, F1={m.overall.f1:.3f}, timings={res.timings}")
    return dt, m


def test_vector_large(large):
    out, name = large
    dt, m = _run(out / f"{name}.svg", out / f"{name}_gt.json", "vector")
    assert dt < VECTOR_LIMIT_S
    assert m.overall.f1 >= 0.92


def test_raster_large(large):
    out, name = large
    dt, m = _run(out / f"{name}.png", out / f"{name}_coco.json", "raster")
    assert dt < RASTER_LIMIT_S
