"""3.2 Якоря примитивов и ключи пар, инвариантные к подобию. Ядро (Opus)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from symspot.core.config import Config
from symspot.core.types import Primitive

KIND_SEGMENT = 0
KIND_ARC = 1
KIND_CIRCLE = 2
KIND_TEXT = 3


@dataclass
class Anchors:
    """Якоря примитивов (P штук).

    ``kind`` (P,) — KIND_*; ``pos`` (P, 2) — середина отрезка / центр дуги / центр рамки
    текста; ``dir`` (P,) — направление отрезка (по модулю π), направление на середину дуги
    от центра (ориентированное), угол базовой линии текста; для окружности NaN;
    ``size`` (P,) — длина отрезка / радиус / диагональ рамки текста;
    ``extent`` (P,) — угловой размах дуги (0 для прочих);
    ``text_hash`` (P,) — хеш нормализованной строки текста (0 для прочих).
    """

    kind: np.ndarray
    pos: np.ndarray
    dir: np.ndarray
    size: np.ndarray
    extent: np.ndarray
    text_hash: np.ndarray

    def __len__(self) -> int:
        return int(len(self.kind))


def compute_anchors(prims: Sequence[Primitive]) -> Anchors:
    """Якоря для списка примитивов."""
    raise NotImplementedError


def pair_keys(
    anchors: Anchors, i: np.ndarray, j: np.ndarray, cfg: Config, spread: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Ключи упорядоченных пар (i[k], j[k]), инвариантные к подобию (без отражения).

    Возвращает (keys int64, row), где row[m] — номер пары k, к которой относится keys[m].
    При ``spread=False`` ровно один ключ на пару. При ``spread=True`` (сторона эталона)
    к каждой паре добавляются ключи соседних ячеек квантования, чтобы значения у границы
    ячейки не терялись. Пары, для которых ключ не определен (обе окружности с общим центром,
    вырожденные), пропускаются.
    """
    raise NotImplementedError
