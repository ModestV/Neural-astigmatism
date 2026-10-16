"""Пространственный индекс кусков чертежа. Задача T5."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

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
    - ``kdtree``: ``scipy.spatial.cKDTree`` по ``samples`` (None, если сэмплов нет);
    - ``sample_step``: float.
    """

    def __init__(self, prims: Sequence[Primitive], sample_step: float) -> None:
        raise NotImplementedError

    def query_bbox(self, bbox: BBox) -> np.ndarray:
        """Индексы (в ``prims``) примитивов, чьи рамки пересекают ``bbox``, по возрастанию."""
        raise NotImplementedError

    def texts_in_bbox(self, bbox: BBox) -> list[TextPrim]:
        """TextPrim, чьи центры рамок лежат в ``bbox``."""
        raise NotImplementedError

    def samples_in_radius(self, pts: np.ndarray, r: float) -> list[np.ndarray]:
        """Для каждой точки (N, 2) — индексы сэмплов ближе ``r``."""
        raise NotImplementedError

    def nearest_sample(self, pts: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
        """Ближайший сэмпл для каждой точки: (dist (N,), idx (N,)); при отсутствии ближе
        ``max_dist`` — dist = inf, idx = -1."""
        raise NotImplementedError
