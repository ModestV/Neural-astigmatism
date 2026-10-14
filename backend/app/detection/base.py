"""Интерфейс алгоритма поиска условных обозначений.

Чтобы подключить свой алгоритм:
  1. создайте класс-наследник Detector (пример — stub.py);
  2. добавьте его в DETECTORS в detection/__init__.py;
  3. запустите бэкенд с переменной окружения SYMSEARCH_DETECTOR=<имя>.

Алгоритм на Julia можно вызывать отсюда же — через juliacall или как отдельный процесс.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..models import BBox, Detection, Drawing, SearchParams


@dataclass(frozen=True)
class PageRef:
    """Страница чертежа со всеми её представлениями.

    Координаты (BBox, Detection) — в единицах страницы, см. app/converters.py.
    """

    drawing: Drawing
    index: int
    width: float
    height: float
    svg_path: Path
    """Страница в SVG — ровно то, что видит пользователь."""
    pdf_path: Path | None
    """Векторный PDF в тех же координатах (для PDF, DXF и DWG; для SVG — None)."""
    source_path: Path
    """Исходный загруженный файл."""


@dataclass(frozen=True)
class Template:
    """Образец для поиска — прямоугольная область на странице какого-либо чертежа."""

    page: PageRef
    bbox: BBox
    symbol_id: str | None = None
    """id в библиотеке; None — фрагмент, выделенный без сохранения."""
    name: str | None = None


class Detector(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def detect(
        self, target: PageRef, templates: Sequence[Template], params: SearchParams
    ) -> list[Detection]:
        """Найти на странице target все вхождения образцов templates."""
