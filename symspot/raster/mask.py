"""Маски пикселей экземпляров и RLE. Фаза 5, Opus."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.types import Instance, Query
from symspot.raster.preprocess import RasterPrep

# Допуск угла для пикселей маски: шире, чем в ICP (направление по пикселям шумнее).
MASK_ANGLE_FACTOR = 2.0


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
    направлением -> RLE в рамке экземпляра.

    Допуск = ``cover_tol`` * диагональ экземпляра + толщина линии. Пиксели без направления
    (стыки, концы линий) принимаются по одному расстоянию.
    """
    h, w = prep.ink.shape
    T = instance.T
    tol = cfg.cover_tol * T.s + prep.stroke_width
    x0 = max(0, int(math.floor(instance.bbox[0] - tol)))
    y0 = max(0, int(math.floor(instance.bbox[1] - tol)))
    x1 = min(w, int(math.ceil(instance.bbox[2] + tol)))
    y1 = min(h, int(math.ceil(instance.bbox[3] + tol)))
    if x1 <= x0 or y1 <= y0 or len(query.samples) == 0:
        return rle_encode(np.zeros((0, 0), dtype=bool), (x0, y0))
    ink = prep.ink[y0:y1, x0:x1]
    ys, xs = np.nonzero(ink)
    mask = np.zeros(ink.shape, dtype=bool)
    if len(xs):
        P = T.apply(query.samples)
        Pt = T.apply_vec(query.tangents)
        d, j = cKDTree(P).query(np.stack([xs + x0 + 0.5, ys + y0 + 0.5], axis=1), k=1)
        ok = d <= tol
        ang = prep.direction[ys + y0, xs + x0]
        has_dir = np.isfinite(ang)
        cos_min = math.cos(math.radians(MASK_ANGLE_FACTOR * cfg.icp_angle_deg))
        cosv = np.abs(np.cos(np.nan_to_num(ang)) * Pt[j, 0] + np.sin(np.nan_to_num(ang)) * Pt[j, 1])
        ok &= ~has_dir | (cosv >= cos_min) | (d <= prep.stroke_width)
        mask[ys[ok], xs[ok]] = True
    return rle_encode(mask, (x0, y0))
