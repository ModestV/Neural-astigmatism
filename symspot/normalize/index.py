"""Пространственный индекс кусков чертежа. Задача T5."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import shapely
from scipy.spatial import cKDTree

from symspot.core.geometry_utils import prim_bbox, sample_primitives
from symspot.core.types import BBox, Primitive, TextPrim


class PrimIndex:
    """STRtree по рамкам примитивов + cKDTree по сэмплам.

    Атрибуты (заполняются в ``__init__``):
    - ``prims``: list[Primitive] — все примитивы в исходном порядке (вкл. TextPrim);
    - ``ids``: np.ndarray[int] (P,) — ``prims[i].id``;
    - ``id_to_idx``: dict[int, int];
    - ``bboxes``: np.ndarray (P, 4);
    - ``samples``, ``tangents``: (M, 2); ``sample_prim``: (M,) индекс в ``prims``
      (``geometry_utils.sample_primitives`` с шагом ``sample_step``; текст сэмплов не дает);
    - ``sample_start``: np.ndarray (P + 1,) — сэмплы примитива i лежат в
      ``samples[sample_start[i]:sample_start[i + 1]]``;
    - ``kdtree``: ``scipy.spatial.cKDTree`` по ``samples`` (None, если сэмплов нет);
    - ``sample_step``: float.
    """

    def __init__(self, prims: Sequence[Primitive], sample_step: float) -> None:
        self.prims: list[Primitive] = list(prims)
        self.sample_step = float(sample_step)
        n = len(self.prims)
        self.ids = np.array([p.id for p in self.prims], dtype=np.int64)
        self.id_to_idx: dict[int, int] = {p.id: i for i, p in enumerate(self.prims)}
        self.bboxes = (
            np.array([prim_bbox(p) for p in self.prims], dtype=float)
            if n
            else np.empty((0, 4), dtype=float)
        )
        self.samples, self.tangents, self.sample_prim = sample_primitives(
            self.prims, self.sample_step
        )
        counts = np.bincount(self.sample_prim, minlength=n) if n else np.zeros(0, dtype=np.int64)
        self.sample_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.kdtree: cKDTree | None = cKDTree(self.samples) if len(self.samples) else None
        self._text_idx = np.array(
            [i for i, p in enumerate(self.prims) if isinstance(p, TextPrim)], dtype=np.int64
        )
        if n:
            b = self.bboxes
            self._tree: shapely.STRtree | None = shapely.STRtree(
                shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3])
            )
        else:
            self._tree = None

    def query_bbox(self, bbox: BBox) -> np.ndarray:
        """Индексы (в ``prims``) примитивов, чьи рамки пересекают ``bbox``, по возрастанию."""
        if self._tree is None:
            return np.empty(0, dtype=np.int64)
        res = self._tree.query(shapely.box(bbox[0], bbox[1], bbox[2], bbox[3]), "intersects")
        return np.sort(np.asarray(res, dtype=np.int64))

    def texts_in_bbox(self, bbox: BBox) -> list[TextPrim]:
        """TextPrim, чьи центры рамок лежат в ``bbox``."""
        out: list[TextPrim] = []
        for i in self._text_idx:
            b = self.bboxes[i]
            cx = 0.5 * (b[0] + b[2])
            cy = 0.5 * (b[1] + b[3])
            if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                p = self.prims[int(i)]
                assert isinstance(p, TextPrim)
                out.append(p)
        return out

    def samples_in_radius(self, pts: np.ndarray, r: float) -> list[np.ndarray]:
        """Для каждой точки (N, 2) — индексы сэмплов ближе ``r``."""
        q = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.kdtree is None:
            return [np.empty(0, dtype=np.int64) for _ in range(len(q))]
        found = self.kdtree.query_ball_point(q, r)
        return [np.sort(np.asarray(f, dtype=np.int64)) for f in found]

    def nearest_sample(self, pts: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
        """Ближайший сэмпл для каждой точки: (dist (N,), idx (N,)); при отсутствии ближе
        ``max_dist`` — dist = inf, idx = -1."""
        q = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.kdtree is None or len(q) == 0:
            return np.full(len(q), np.inf), np.full(len(q), -1, dtype=np.int64)
        dist, idx = self.kdtree.query(q, k=1, distance_upper_bound=max_dist)
        idx = np.asarray(idx, dtype=np.int64)
        miss = idx >= len(self.samples)
        dist = np.where(miss, np.inf, dist)
        idx = np.where(miss, -1, idx)
        return dist, idx
