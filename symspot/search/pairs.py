"""3.2б Таблица пар кусков чертежа. Ядро (Opus)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symspot.core.config import Config
from symspot.normalize.index import PrimIndex
from symspot.query.descriptors import Anchors


@dataclass
class PairTable:
    """Пары кусков чертежа (a, b) с якорями ближе ``radius``; ``keys`` отсортированы.

    ``a``, ``b`` — индексы в ``index.prims``; ``anchors`` — якоря всех ``index.prims``.
    """

    keys: np.ndarray
    a: np.ndarray
    b: np.ndarray
    anchors: Anchors
    radius: float

    @staticmethod
    def build(index: PrimIndex, radius: float, cfg: Config) -> PairTable:
        raise NotImplementedError

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Для массива ключей -> (k, row): keys[k] совпал со строкой таблицы row."""
        raise NotImplementedError
