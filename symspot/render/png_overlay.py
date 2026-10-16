"""PNG-оверлей. Задача T4."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from symspot.core.types import Drawing, Result
from symspot.raster.mask import rle_decode
from symspot.render._cairo import load_cairosvg
from symspot.render._common import RASTER_LIGHTEN_B, RASTER_LIGHTEN_K
from symspot.render.palette import color_for
from symspot.render.svg_overlay import build_svg


def _render_raster(drawing: Drawing, result: Result) -> np.ndarray:
    if drawing.image is None:
        raise ValueError("растровый чертеж без изображения")
    gray = np.asarray(drawing.image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    light = np.clip(RASTER_LIGHTEN_K * gray.astype(np.float32) + RASTER_LIGHTEN_B, 0, 255)
    canvas = cv2.cvtColor(light.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    H, W = canvas.shape[:2]
    for inst in result.instances:
        r, g, b = color_for(inst.legend_id)
        if inst.mask_rle is not None:
            mask, (ox, oy) = rle_decode(inst.mask_rle)
            h, w = mask.shape
            x0, y0 = max(ox, 0), max(oy, 0)
            x1, y1 = min(ox + w, W), min(oy + h, H)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = mask[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
            canvas[y0:y1, x0:x1][sub] = (b, g, r)
        else:
            bx0, by0, bx1, by1 = (int(round(v)) for v in inst.bbox)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (b, g, r), 1)
    return canvas


def render_png(drawing: Drawing, result: Result, out_path: str | Path, scale: float = 2.0) -> Path:
    """Растр: исходное изображение светлее + пиксели масок цветом экземпляра.
    Вектор: растеризация SVG-оверлея (cairosvg) с масштабом ``scale`` px на единицу."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if drawing.kind == "raster":
        if not cv2.imwrite(str(out), _render_raster(drawing, result)):
            raise OSError(f"не удалось записать PNG: {out}")
        return out
    svg = build_svg(drawing, result)
    load_cairosvg().svg2png(bytestring=svg.encode("utf-8"), write_to=str(out), scale=scale)
    return out
