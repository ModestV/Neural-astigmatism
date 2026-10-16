"""Проверка гипотез на растре (переиспользует ядро фазы 3). Фаза 5, Opus.

Растр сводится к тем же сущностям, что и вектор: «сэмплы чертежа» = пиксели скелета туши с
направлением линии, поэтому ICP, покрытие и соревнование из ``verify/`` работают без
изменений через ``RasterIndex`` (тот же интерфейс, что у ``PrimIndex``, без примитивов).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.geometry_utils import bbox_center, bbox_diag
from symspot.core.types import BBox, Primitive, Query, QuerySpec, Similarity, TextPrim
from symspot.raster.preprocess import RasterPrep


class RasterIndex:
    """Индекс по пикселям скелета (координаты центров пикселей, ось Y вниз)."""

    def __init__(self, prep: RasterPrep, sample_step: float = 1.0) -> None:
        ys, xs = np.nonzero(prep.skeleton)
        pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(float)
        ang = prep.direction[ys, xs].astype(float)
        ang = np.where(np.isfinite(ang), ang, 0.0)
        self.prims: list[Primitive] = []
        self.ids = np.empty(0, dtype=np.int64)
        self.id_to_idx: dict[int, int] = {}
        self.bboxes = np.empty((0, 4))
        self.samples = pts
        self.tangents = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.sample_prim = np.zeros(len(pts), dtype=np.int64)
        self.sample_start = np.zeros(1, dtype=np.int64)
        self.sample_step = float(sample_step)
        self.kdtree = cKDTree(pts) if len(pts) else None
        self._symspot_half_width = np.full(len(pts), 0.5 * prep.stroke_width)

    def query_bbox(self, bbox: BBox) -> np.ndarray:
        return np.empty(0, dtype=np.int64)

    def texts_in_bbox(self, bbox: BBox) -> list[TextPrim]:
        return []

    def samples_in_radius(self, pts: np.ndarray, r: float) -> list[np.ndarray]:
        if self.kdtree is None:
            return [np.empty(0, dtype=np.int64) for _ in range(len(pts))]
        return [np.asarray(x, dtype=np.int64) for x in self.kdtree.query_ball_point(pts, r)]

    def nearest_sample(self, pts: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(pts)
        if self.kdtree is None:
            return np.full(n, np.inf), np.full(n, -1, dtype=np.int64)
        d, i = self.kdtree.query(pts, k=1, distance_upper_bound=max_dist)
        i = np.where(np.isfinite(d), i, -1)
        return d, i


def extract_raster_query(spec: QuerySpec, prep: RasterPrep, cfg: Config) -> Query:
    """Эталон из скелета в рамке: сэмплы = пиксели скелета (прореженные), касательные из
    ``prep.direction``, нормализация как в векторной ветке."""
    x0, y0, x1, y1 = (
        int(math.floor(spec.bbox[0])),
        int(math.floor(spec.bbox[1])),
        int(math.ceil(spec.bbox[2])),
        int(math.ceil(spec.bbox[3])),
    )
    h, w = prep.skeleton.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    sk = prep.skeleton[y0:y1, x0:x1]
    ys, xs = np.nonzero(sk)
    if len(xs) < 3:
        raise ValueError(f"эталон {spec.legend_id}: в рамке {spec.bbox} нет туши")
    pts = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], axis=1).astype(float)
    ang = prep.direction[ys + y0, xs + x0].astype(float)
    ang = np.where(np.isfinite(ang), ang, 0.0)
    diag = bbox_diag(spec.bbox)
    cx, cy = bbox_center(spec.bbox)
    frame = Similarity(diag, 0.0, cx, cy)
    norm = frame.inverse().apply(pts)
    # прореживание сеткой с шагом sample_step (в нормализованных единицах)
    step = cfg.sample_step
    cell = np.floor(norm / step).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    first.sort()
    norm = norm[first]
    ang = ang[first]
    tans = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    x0b, y0b, x1b, y1b = spec.bbox
    nbox = ((x0b - cx) / diag, (y0b - cy) / diag, (x1b - cx) / diag, (y1b - cy) / diag)
    return Query(
        legend_id=spec.legend_id,
        prims=[],
        frame=frame,
        samples=norm,
        tangents=tans,
        weights=np.ones(len(norm)),
        sample_prim=np.zeros(len(norm), dtype=np.int64),
        diag=diag,
        label=spec.label,
        box=nbox,
    )
