from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from symspot.core.config import Config
from symspot.raster.preprocess import RasterPrep, preprocess

THICK = 3


def _canvas(h: int = 200, w: int = 300) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _mod_pi_dist(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % math.pi
    return np.minimum(d, math.pi - d)


def _line_image() -> np.ndarray:
    img = _canvas()
    cv2.line(img, (20, 50), (250, 50), 0, THICK)  # горизонталь
    cv2.line(img, (150, 80), (150, 180), 0, THICK)  # вертикаль
    cv2.rectangle(img, (270, 150), (271, 151), 0, -1)  # мусор 2x2 (< min_blob_px)
    return img


def _true_width(img: np.ndarray) -> int:
    """Фактическая толщина горизонтальной линии (cv2 рисует thickness=3 шире 3 px)."""
    return int((img[:, 100] < 128).sum())


@pytest.fixture(scope="module")
def lines_prep() -> RasterPrep:
    return preprocess(_line_image(), Config())


def test_shapes_and_types(lines_prep: RasterPrep) -> None:
    p = lines_prep
    assert p.gray.shape == p.ink.shape == p.skeleton.shape == p.dist.shape == p.direction.shape
    assert p.gray.dtype == np.uint8
    assert p.ink.dtype == bool and p.skeleton.dtype == bool
    assert p.dist.dtype == np.float32 and p.direction.dtype == np.float32
    assert p.deskew_angle == 0.0


def test_skeleton_thin_and_inside_ink(lines_prep: RasterPrep) -> None:
    p = lines_prep
    assert (p.skeleton & ~p.ink).sum() == 0
    # у горизонтальной линии в каждом столбце не больше 1-2 пикселей скелета
    per_col = p.skeleton[40:60, 40:230].sum(axis=0)
    assert per_col.max() <= 2 and per_col.min() >= 1
    width = _true_width(_line_image())
    assert p.skeleton.sum() < 1.3 * p.ink.sum() / width


def test_stroke_width_estimate(lines_prep: RasterPrep) -> None:
    assert lines_prep.stroke_width == pytest.approx(_true_width(_line_image()), abs=0.6)


def test_dist_zero_on_skeleton(lines_prep: RasterPrep) -> None:
    p = lines_prep
    assert np.all(p.dist[p.skeleton] == 0.0)
    assert p.dist[~p.skeleton].min() > 0.0
    assert p.dist[10, 100] == pytest.approx(40.0, abs=1.5)  # до линии y=50


def test_direction_horizontal_and_vertical(lines_prep: RasterPrep) -> None:
    p = lines_prep
    h = p.direction[50, 60:230]
    v = p.direction[90:170, 150]
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(v))
    assert _mod_pi_dist(h, 0.0).max() < 0.1
    assert _mod_pi_dist(v, math.pi / 2).max() < 0.1
    known = p.direction[np.isfinite(p.direction)]
    assert np.all((known >= 0) & (known < math.pi + 1e-6))


def test_direction_nan_outside_ink(lines_prep: RasterPrep) -> None:
    p = lines_prep
    assert np.all(np.isnan(p.direction[~p.ink]))
    assert np.all(np.isfinite(p.direction[p.ink]))


def test_small_blobs_removed(lines_prep: RasterPrep) -> None:
    assert not lines_prep.ink[148:154, 268:274].any()
    assert not lines_prep.skeleton[148:154, 268:274].any()


def test_diagonal_line_direction() -> None:
    img = _canvas()
    cv2.line(img, (20, 20), (180, 180), 0, THICK)
    p = preprocess(img, Config())
    d = p.direction[p.skeleton][20:-20]
    assert _mod_pi_dist(d, math.pi / 4).max() < 0.15


def test_blank_image() -> None:
    p = preprocess(_canvas(), Config())
    assert not p.ink.any() and not p.skeleton.any()
    assert np.all(np.isnan(p.direction))
    assert np.all(np.isfinite(p.dist))
    assert p.stroke_width >= 1.0


def test_color_input_is_converted() -> None:
    gray = _canvas()
    cv2.line(gray, (20, 50), (250, 50), 0, THICK)
    rgb = np.stack([gray] * 3, axis=2)
    p = preprocess(rgb, Config())
    assert p.gray.ndim == 2
    assert np.array_equal(p.ink, preprocess(gray, Config()).ink)


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        preprocess(np.zeros(10, dtype=np.uint8), Config())


def test_deskew_estimates_angle_without_rotating() -> None:
    img = _canvas(600, 800)
    ang = math.radians(3.0)
    # рамка и линии, повернутые на 3 градуса
    for (x0, y0), (x1, y1) in [((100, 100), (700, 100)), ((100, 500), (700, 500))]:
        c, s = math.cos(ang), math.sin(ang)
        p0 = (int(x0 * c - y0 * s + 40), int(x0 * s + y0 * c))
        p1 = (int(x1 * c - y1 * s + 40), int(x1 * s + y1 * c))
        cv2.line(img, p0, p1, 0, THICK)
    off = preprocess(img, Config(deskew=False))
    on = preprocess(img, Config(deskew=True))
    assert off.deskew_angle == 0.0
    assert on.deskew_angle == pytest.approx(ang, abs=math.radians(0.7))
    assert np.array_equal(on.ink, off.ink) and np.array_equal(on.skeleton, off.skeleton)
