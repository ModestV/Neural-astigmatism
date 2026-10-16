"""Чтение растра PNG/JPG."""

from __future__ import annotations

from pathlib import Path

import cv2

from symspot.core.types import Drawing


def read_raster(path: str | Path) -> Drawing:
    """Прочитать изображение в ``Drawing(kind="raster")`` с grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"не удалось прочитать изображение {path}")
    h, w = img.shape
    return Drawing(
        kind="raster",
        width=float(w),
        height=float(h),
        units_per_px=1.0,
        image=img,
        source_path=str(path),
    )
