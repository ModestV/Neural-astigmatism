"""Растровая синтетика для тестов chamfer: листы с копиями простых символов."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from symspot.core.config import Config
from symspot.core.types import QuerySpec, Similarity

THICK = 2

# символы в локальных координатах: круги (cx, cy, r) и ломаные
SYMBOLS: dict[str, dict] = {
    "slash_circle": {
        "circles": [(0.0, 0.0, 20.0)],
        "lines": [[(-14.1, 14.1), (14.1, -14.1)]],
        "symmetric": True,
    },
    "tri_stem": {
        "circles": [],
        "lines": [
            [(-20.0, 15.0), (25.0, 15.0), (-5.0, -20.0), (-20.0, 15.0)],
            [(-5.0, -20.0), (-5.0, -38.0)],
            [(-5.0, -38.0), (8.0, -38.0)],
        ],
        "symmetric": False,
    },
}


@dataclass(frozen=True)
class Copy:
    name: str
    scale: float
    theta_deg: float
    mirror: bool
    xy: tuple[float, float]  # куда уходит начало локальных координат


def local_bbox(name: str) -> tuple[float, float, float, float]:
    pts = []
    for cx, cy, r in SYMBOLS[name]["circles"]:
        pts += [(cx - r, cy - r), (cx + r, cy + r)]
    for ln in SYMBOLS[name]["lines"]:
        pts += list(ln)
    a = np.array(pts)
    return (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())


def draw_copy(img: np.ndarray, c: Copy) -> tuple[float, float, float, float]:
    """Нарисовать копию, вернуть ее рамку (x0, y0, x1, y1) на листе."""
    T = Similarity(c.scale, math.radians(c.theta_deg), c.xy[0], c.xy[1], c.mirror)
    sym = SYMBOLS[c.name]
    for cx, cy, r in sym["circles"]:
        p = T.apply(np.array([cx, cy]))
        cv2.circle(img, (round(p[0]), round(p[1])), round(r * c.scale), 0, THICK, cv2.LINE_AA)
    for ln in sym["lines"]:
        p = np.round(T.apply(np.array(ln))).astype(np.int32)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], False, 0, THICK, cv2.LINE_AA)
    x0, y0, x1, y1 = local_bbox(c.name)
    corners = T.apply(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    )


def local_center(name: str) -> np.ndarray:
    x0, y0, x1, y1 = local_bbox(name)
    return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])


def center_of(c: Copy) -> np.ndarray:
    """Положение центра локальной рамки символа на листе."""
    T = Similarity(c.scale, math.radians(c.theta_deg), c.xy[0], c.xy[1], c.mirror)
    return T.apply(local_center(c.name))


def make_sheet(
    copies: list[Copy], legend: dict[str, tuple[float, float]], size=(1500, 1000), clutter=True
) -> tuple[np.ndarray, dict[str, QuerySpec]]:
    """Лист (серый uint8, белый фон) с копиями и «легендой» (эталоны без поворота)."""
    w, h = size
    img = np.full((h, w), 255, dtype=np.uint8)
    if clutter:
        cv2.line(img, (30, 480), (w - 30, 500), 0, THICK, cv2.LINE_AA)
        cv2.line(img, (700, 30), (720, h - 30), 0, THICK, cv2.LINE_AA)
        cv2.rectangle(img, (40, 40), (w - 40, h - 40), 0, THICK, cv2.LINE_AA)
    specs: dict[str, QuerySpec] = {}
    for name, xy in legend.items():
        bb = draw_copy(img, Copy(name, 1.0, 0.0, False, xy))
        pad = 4.0
        specs[name] = QuerySpec(name, (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad))
    for c in copies:
        draw_copy(img, c)
    return img, specs


DEFAULT_CFG = Config
