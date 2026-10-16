"""HTML-отчет. Задача T4."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import Drawing, Result


def render_html(drawing: Drawing, result: Result, out_path: str | Path) -> Path:
    """Один самодостаточный HTML: встроенный SVG-оверлей с zoom/pan, таблица экземпляров
    (id, класс, score, coverage, status), фильтр по классу, клик по строке -> подсветка и
    центрирование экземпляра, отдельный блок ``ambiguous``, тайминги."""
    raise NotImplementedError
