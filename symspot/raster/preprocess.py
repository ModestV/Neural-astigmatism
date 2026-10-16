"""Предобработка растра. Фаза 5."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize

from symspot.core.config import Config

# Строк в полосе при вычислении направления (память на листах 7200x4800).
_BAND_ROWS = 1024
# Поле полосы: больше радиуса сглаживания структурного тензора.
_BAND_MARGIN = 16


@dataclass
class RasterPrep:
    """Результат предобработки (все массивы размера изображения H x W).

    ``ink`` — бинарная тушь (True = линия) после Sauvola и удаления мусора;
    ``skeleton`` — скелет туши (bool); ``dist`` — float32 расстояние (px) до ближайшего
    пикселя скелета; ``direction`` — float32 направление линии (радианы, по модулю π) в каждом
    пикселе туши (по структурному тензору/градиенту), NaN вне туши;
    ``stroke_width`` — оценка медианной толщины линии (px); ``deskew_angle`` — примененный
    поворот (радианы, 0 если не было).
    """

    gray: np.ndarray
    ink: np.ndarray
    skeleton: np.ndarray
    dist: np.ndarray
    direction: np.ndarray
    stroke_width: float
    deskew_angle: float = 0.0


def to_gray(image: np.ndarray) -> np.ndarray:
    """Изображение (H, W) или (H, W, 3/4) любого типа -> uint8 (H, W)."""
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"ожидалось изображение (H, W) или (H, W, C), получено {img.shape}")
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    if img.dtype == bool:
        return np.where(img, 255, 0).astype(np.uint8)
    arr = img.astype(np.float64)
    if arr.size and arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """Тушь = темные пиксели: Sauvola, плюс отсечение светлого фона и явно черного."""
    window = max(3, int(cfg.sauvola_window) | 1)
    thr = threshold_sauvola(gray, window_size=window, k=cfg.sauvola_k)
    ink = (gray < thr) & (gray <= cfg.raster_white)
    ink |= gray <= cfg.raster_black
    return ink


def remove_small_blobs(ink: np.ndarray, min_px: int) -> np.ndarray:
    """Удалить связные компоненты (8-связность) площадью меньше ``min_px``."""
    if min_px <= 1 or not ink.any():
        return ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


def estimate_skew(ink: np.ndarray, max_side: int = 2000) -> float:
    """Угол перекоса (радианы, в (-π/4, π/4]) по длинным линиям (Hough). 0, если линий нет."""
    h, w = ink.shape
    f = max(1.0, max(h, w) / max_side)
    small = cv2.resize(
        ink.astype(np.uint8) * 255,
        (max(1, int(w / f)), max(1, int(h / f))),
        interpolation=cv2.INTER_AREA,
    )
    small = (small > 64).astype(np.uint8) * 255
    min_len = int(0.1 * min(small.shape))
    lines = cv2.HoughLinesP(
        small,
        1,
        math.pi / 720,
        threshold=max(20, min_len // 2),
        minLineLength=min_len,
        maxLineGap=3,
    )
    if lines is None or len(lines) == 0:
        return 0.0
    ls = np.asarray(lines, dtype=float).reshape(-1, 4)
    dx, dy = ls[:, 2] - ls[:, 0], ls[:, 3] - ls[:, 1]
    length = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    folded = (ang + math.pi / 4) % (math.pi / 2) - math.pi / 4  # к ближайшей оси
    order = np.argsort(folded)
    cum = np.cumsum(length[order])
    k = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(folded[order][min(k, len(order) - 1)])


def line_direction(ink: np.ndarray, sigma_grad: float = 1.0, sigma_win: float = 2.0) -> np.ndarray:
    """Направление линии (по модулю π) в пикселях туши, NaN вне туши.

    Структурный тензор сглаженной маски туши: направление градиента -> перпендикуляр.
    """
    h, w = ink.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    if not ink.any():
        return out
    for y0 in range(0, h, _BAND_ROWS):
        y1 = min(h, y0 + _BAND_ROWS)
        a0, a1 = max(0, y0 - _BAND_MARGIN), min(h, y1 + _BAND_MARGIN)
        band = ink[a0:a1]
        if not band[y0 - a0 : y1 - a0].any():
            continue
        f = cv2.GaussianBlur(band.astype(np.float32), (0, 0), sigma_grad)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_win)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_win)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_win)
        r0, r1 = y0 - a0, y1 - a0
        m = band[r0:r1]
        ys, xs = np.nonzero(m)
        grad_ang = 0.5 * np.arctan2(
            2.0 * jxy[r0:r1][ys, xs], jxx[r0:r1][ys, xs] - jyy[r0:r1][ys, xs]
        )
        out[y0 + ys, xs] = np.mod(grad_ang + math.pi / 2, math.pi)
    return out


def preprocess(image: np.ndarray, cfg: Config) -> RasterPrep:
    """Серое -> Sauvola (``sauvola_window``, ``sauvola_k``) -> удаление компонент меньше
    ``min_blob_px`` -> опционально deskew (Hough, при ``cfg.deskew``) -> скелет ->
    карта расстояний -> направление.

    Координаты всех массивов остаются в системе исходного изображения: при ``cfg.deskew``
    угол только оценивается и записывается в ``deskew_angle`` (массивы не поворачиваются).
    """
    gray = to_gray(image)
    ink = remove_small_blobs(binarize(gray, cfg), cfg.min_blob_px)
    angle = estimate_skew(ink) if cfg.deskew else 0.0

    skeleton = skeletonize(ink) if ink.any() else np.zeros_like(ink)
    skeleton = np.ascontiguousarray(skeleton, dtype=bool)

    if skeleton.any():
        dist = cv2.distanceTransform((~skeleton).astype(np.uint8), cv2.DIST_L2, 5)
        dt_ink = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
        # нижняя оценка ширины: ось нечетной линии на расстоянии d от фона -> 2d - 1;
        # для четной ширины она занижена на 1, поэтому верхняя граница = нижняя + 1,
        # а внутри берется площадь / длина скелета
        low = float(np.median(2.0 * dt_ink[skeleton] - 1.0))
        area_ratio = float(ink.sum()) / float(skeleton.sum())
        width = min(max(area_ratio, low), low + 1.0)
    else:
        dist = np.full(gray.shape, np.float32(max(gray.shape)), dtype=np.float32)
        width = 1.0

    return RasterPrep(
        gray=gray,
        ink=ink,
        skeleton=skeleton,
        dist=dist.astype(np.float32, copy=False),
        direction=line_direction(ink),
        stroke_width=max(1.0, width),
        deskew_angle=angle,
    )
