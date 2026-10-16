"""PNG-оверлей. Задача T4."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import Drawing, Result


def render_png(drawing: Drawing, result: Result, out_path: str | Path, scale: float = 2.0) -> Path:
    """Растр: исходное изображение светлее + пиксели масок цветом экземпляра.
    Вектор: растеризация SVG-оверлея (cairosvg) с масштабом ``scale`` px на единицу."""
    raise NotImplementedError
