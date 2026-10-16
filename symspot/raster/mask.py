"""Маски пикселей экземпляров и RLE. Фаза 5, Opus."""

from __future__ import annotations

import numpy as np

from symspot.core.config import Config
from symspot.core.types import Instance, Query
from symspot.raster.preprocess import RasterPrep


def rle_encode(mask: np.ndarray, origin: tuple[int, int] = (0, 0)) -> dict:
    """Маска (h, w) bool -> {"origin": [x0, y0], "size": [h, w], "counts": [...]}.

    counts — длины серий в порядке строк (row-major), первая серия — нули (может быть 0).
    """
    m = np.ascontiguousarray(mask, dtype=bool).ravel()
    h, w = mask.shape
    if m.size == 0:
        return {"origin": [int(origin[0]), int(origin[1])], "size": [int(h), int(w)], "counts": []}
    change = np.flatnonzero(m[1:] != m[:-1]) + 1
    bounds = np.concatenate([[0], change, [m.size]])
    runs = np.diff(bounds).tolist()
    if m[0]:
        runs = [0, *runs]
    return {"origin": [int(origin[0]), int(origin[1])], "size": [int(h), int(w)], "counts": runs}


def rle_decode(rle: dict) -> tuple[np.ndarray, tuple[int, int]]:
    """RLE -> (маска (h, w) bool, origin (x0, y0))."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    val = False
    for c in rle["counts"]:
        if val:
            flat[pos : pos + c] = True
        pos += c
        val = not val
    x0, y0 = rle["origin"]
    return flat.reshape(h, w), (int(x0), int(y0))


def instance_mask(instance: Instance, query: Query, prep: RasterPrep, cfg: Config) -> dict:
    """Пиксели туши ближе допуска к трансформированному эталону с согласованным
    направлением -> RLE в рамке экземпляра."""
    raise NotImplementedError
