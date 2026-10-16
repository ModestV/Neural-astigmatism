"""Элементы чертежа синтетики: примитив + параметры вывода в SVG."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from symspot.core.types import BBox, Primitive, TextPrim

TEXT_WIDTH_FACTOR = 0.6  # оценка ширины символа текста в долях кегля
BEZIER_KAPPA = 0.5522847498307936  # константа четырех кубических Безье для окружности


@dataclass
class Element:
    """Один элемент SVG: путь (отрезок, дуга, окружность) или текст.

    ``prim`` в координатах листа. ``font_size`` > 0 только для текста.
    ``bezier`` = полную окружность записать четырьмя кубическими Безье, а не командами A.
    """

    id: str
    prim: Primitive
    width: float = 0.5
    font_size: float = 0.0
    bezier: bool = False


def text_prim(pid: int, s: str, cx: float, cy: float, fs: float, angle: float = 0.0) -> TextPrim:
    """Текст с центром рамки (cx, cy); рамка = выровненная по осям оценочная рамка."""
    w = TEXT_WIDTH_FACTOR * fs * len(s)
    h = fs
    c, si = math.cos(angle), math.sin(angle)
    hx = abs(c) * w / 2 + abs(si) * h / 2
    hy = abs(si) * w / 2 + abs(c) * h / 2
    return TextPrim(
        id=pid, text=s, bbox=(cx - hx, cy - hy, cx + hx, cy + hy), parent_id=pid, angle=angle
    )


def box_intersects(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class BoxIndex:
    """Набор прямоугольников с векторной проверкой пересечения (для размещения)."""

    def __init__(self) -> None:
        self._a = np.empty((256, 4), dtype=float)
        self._n = 0

    def add(self, b: BBox) -> None:
        if self._n == len(self._a):
            self._a = np.concatenate([self._a, np.empty_like(self._a)])
        self._a[self._n] = b
        self._n += 1

    def hits(self, b: BBox, margin: float = 0.0) -> bool:
        if self._n == 0:
            return False
        a = self._a[: self._n]
        m = (
            (a[:, 0] < b[2] + margin)
            & (a[:, 2] > b[0] - margin)
            & (a[:, 1] < b[3] + margin)
            & (a[:, 3] > b[1] - margin)
        )
        return bool(m.any())
