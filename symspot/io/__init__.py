"""Ввод: чертежи и разметка."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import Drawing

RASTER_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_drawing(path: str | Path, page: int = 0) -> Drawing:
    """Прочитать чертеж по расширению файла."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        from symspot.io.pdf_reader import read_pdf

        return read_pdf(path, page=page)
    if ext == ".svg":
        from symspot.io.svg_reader import read_svg

        return read_svg(path)
    if ext in RASTER_EXT:
        from symspot.io.raster_reader import read_raster

        return read_raster(path)
    raise ValueError(f"неподдерживаемый формат: {ext}")
