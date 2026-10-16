"""Нормализация векторного чертежа: Безье, склейка, планаризация."""

from __future__ import annotations

import dataclasses

from symspot.core.config import Config
from symspot.core.types import Drawing, TextPrim


def normalize_drawing(drawing: Drawing, ref_size: float, cfg: Config) -> Drawing:
    """Штрихи -> склейка -> планаризация. Пороги = доли ``ref_size`` (размер символа)."""
    from symspot.normalize.bezier import parents_to_strokes
    from symspot.normalize.merge import merge_collinear
    from symspot.normalize.planarize import planarize

    if drawing.kind != "vector":
        raise ValueError("normalize_drawing: только для векторных чертежей")
    texts = [p for p in drawing.primitives if isinstance(p, TextPrim)]
    next_id = max([p.id for p in drawing.primitives] + list(drawing.parents) + [-1]) + 1
    strokes = parents_to_strokes(
        drawing.parents, cfg.flatten_tol * ref_size, cfg.arc_fit_tol * ref_size, start_id=next_id
    )
    min_len = cfg.min_piece_len * ref_size
    merged = merge_collinear(strokes, cfg.merge_angle_deg, cfg.merge_gap * ref_size, min_len)
    next_id = max([p.id for p in merged] + [next_id - 1]) + 1
    pieces = planarize(merged, cfg.snap_tol * ref_size, min_len, start_id=next_id)
    texts = [dataclasses.replace(t, stroke_id=t.id) for t in texts]
    return dataclasses.replace(drawing, primitives=[*pieces, *texts], normalized=True)
