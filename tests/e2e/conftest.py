"""Общие фикстуры e2e: синтетика small (seed 0) и прогоны векторного пайплайна."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from symspot.core.config import Config
from symspot.core.types import Drawing, Result
from symspot.pipeline import run
from symspot.synth.export import generate

N_DRAWINGS = 3
SEED = 0


@dataclass
class Run:
    name: str
    ext: str
    gt_path: Path
    result: Result
    drawing: Drawing
    seconds: float


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synth")
    generate(N_DRAWINGS, "small", SEED, d, raster=False)
    return d


@pytest.fixture(scope="session")
def synth_names(synth_dir) -> list[str]:
    return sorted(p.name[: -len("_gt.json")] for p in synth_dir.glob("small_*_gt.json"))


def _run(synth_dir: Path, name: str, ext: str) -> Run:
    gt = synth_dir / f"{name}_gt.json"
    t0 = time.perf_counter()
    result, drawing = run(synth_dir / f"{name}.{ext}", gt, Config())
    dt = time.perf_counter() - t0
    return Run(name, ext, gt, result, drawing, dt)


@pytest.fixture(scope="session")
def runs(synth_dir, synth_names) -> list[Run]:
    """Все прогоны (3 чертежа x SVG, PDF). Первый прогон не считается: прогрев импортов."""
    _run(synth_dir, synth_names[0], "svg")
    return [_run(synth_dir, n, ext) for n in synth_names for ext in ("svg", "pdf")]


@pytest.fixture(scope="session")
def one_run(synth_dir, synth_names) -> Run:
    return _run(synth_dir, synth_names[0], "svg")
