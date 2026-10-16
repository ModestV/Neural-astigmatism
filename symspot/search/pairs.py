"""3.2б Таблица пар кусков чертежа. Ядро (Opus)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.normalize.index import PrimIndex
from symspot.query.descriptors import Anchors, compute_anchors, pair_keys


@dataclass
class PairTable:
    """Пары кусков чертежа (a, b) с якорями ближе ``radius``; ``keys`` отсортированы.

    ``a``, ``b`` — индексы в ``index.prims``; ``anchors`` — якоря всех ``index.prims``.
    Каждая неупорядоченная пара хранится один раз как упорядоченная (a < b); сторона
    эталона перебирает оба порядка.
    """

    keys: np.ndarray
    a: np.ndarray
    b: np.ndarray
    anchors: Anchors
    radius: float

    @staticmethod
    def build(index: PrimIndex, radius: float, cfg: Config) -> PairTable:
        anchors = compute_anchors(index.prims)
        n = len(anchors)
        if n < 2:
            e = np.empty(0, dtype=np.int64)
            return PairTable(e, e, e, anchors, radius)
        tree = cKDTree(anchors.pos)
        k = min(cfg.max_pairs_per_piece + 1, n)
        dist, nbr = tree.query(anchors.pos, k=k, distance_upper_bound=radius)
        src = np.repeat(np.arange(n), k)
        dst = nbr.ravel()
        ok = np.isfinite(dist.ravel()) & (dst < n) & (dst != src)
        src, dst = src[ok], dst[ok]
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
        a, b = pairs[:, 0], pairs[:, 1]
        keys, rows = pair_keys(anchors, a, b, cfg, spread=False)
        a, b = a[rows], b[rows]
        order = np.argsort(keys, kind="stable")
        return PairTable(keys[order], a[order], b[order], anchors, radius)

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Для массива ключей -> (k, row): keys[k] совпал со строкой таблицы row."""
        keys = np.asarray(keys, dtype=np.int64)
        if len(keys) == 0 or len(self.keys) == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e
        lo = np.searchsorted(self.keys, keys, side="left")
        hi = np.searchsorted(self.keys, keys, side="right")
        cnt = hi - lo
        k = np.repeat(np.arange(len(keys)), cnt)
        if len(k) == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e
        starts = np.repeat(lo, cnt)
        offs = np.arange(len(k)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        return k, starts + offs

    def __len__(self) -> int:
        return int(len(self.keys))
