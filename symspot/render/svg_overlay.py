"""SVG-оверлей. Задача T4."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import Drawing, Result


def render_svg(drawing: Drawing, result: Result, out_path: str | Path) -> Path:
    """Исходный чертеж серым (``drawing.parents``, opacity 0.35) + цветной слой: точная
    геометрия кусков ``instance.piece_ids`` (Segment -> line, Arc -> path A, TextPrim ->
    рамка/текст) цветом ``color_for(legend_id)``; ambiguous — пунктир. Для растрового
    чертежа фон = встроенное PNG, слой = маски ``mask_rle``. Размер = размер чертежа,
    viewBox в единицах чертежа. У каждого экземпляра группа ``<g id="inst-<id>">``.
    ``drawing`` должен быть нормализован (куски с теми же id, что в result).
    """
    raise NotImplementedError
