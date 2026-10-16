"""PDF-оверлей. Задача T4."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import Drawing, Result


def render_pdf(drawing: Drawing, result: Result, out_path: str | Path) -> Path:
    """Копия исходной страницы PDF (``drawing.source_path``, ``drawing.page``) + куски
    экземпляров поверх через ``page.new_shape()`` (с учетом rotation страницы:
    координаты чертежа -> координаты страницы через ``page.derotation_matrix``).
    Для не-PDF источника — PDF из SVG-оверлея (cairosvg).
    """
    raise NotImplementedError
