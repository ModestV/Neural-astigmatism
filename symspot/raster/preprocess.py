"""Предобработка растра. Фаза 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symspot.core.config import Config


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


def preprocess(image: np.ndarray, cfg: Config) -> RasterPrep:
    """Серое -> Sauvola (``sauvola_window``, ``sauvola_k``) -> удаление компонент меньше
    ``min_blob_px`` -> опционально deskew (Hough, при ``cfg.deskew``) -> скелет ->
    карта расстояний -> направление."""
    raise NotImplementedError
