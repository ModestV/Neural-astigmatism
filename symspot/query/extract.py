"""3.1 Извлечение эталона из рамки. Ядро (Opus)."""

from __future__ import annotations

import numpy as np

from symspot.core.config import Config
from symspot.core.geometry_utils import (
    bbox_center,
    bbox_diag,
    bbox_expand,
    prim_length,
    sample_primitive,
    sample_primitives,
    transform_prim,
)
from symspot.core.types import BBox, Primitive, Query, QuerySpec, Similarity, TextPrim
from symspot.normalize.index import PrimIndex

# Число сэмплов на кусок при оценке доли длины внутри рамки (точность ~1/INSIDE_SAMPLES).
INSIDE_SAMPLES = 32


def _inside_frac(p: Primitive, box: BBox) -> float:
    L = prim_length(p)
    if L <= 0:
        return 0.0
    pts, _ = sample_primitive(p, L / INSIDE_SAMPLES)
    if len(pts) == 0:
        return 0.0
    ins = (
        (pts[:, 0] >= box[0])
        & (pts[:, 0] <= box[2])
        & (pts[:, 1] >= box[1])
        & (pts[:, 1] <= box[3])
    )
    return float(ins.mean())


def stroke_lengths(index: PrimIndex) -> dict[int, float]:
    """Полная длина каждого штриха (сумма длин его кусков)."""
    out: dict[int, float] = {}
    for p in index.prims:
        if isinstance(p, TextPrim):
            continue
        sid = p.stroke_id if p.stroke_id >= 0 else p.id
        out[sid] = out.get(sid, 0.0) + prim_length(p)
    return out


def make_query(
    legend_id: str,
    prims: list[Primitive],
    frame: Similarity,
    diag: float,
    cfg: Config,
    label: str | None = None,
    piece_ids: list[int] | None = None,
    variant: int = 0,
    box: BBox | None = None,
) -> Query:
    """Query из примитивов, уже переведенных в нормализованные координаты."""
    pts, tan, idx = sample_primitives(prims, cfg.sample_step)
    return Query(
        legend_id=legend_id,
        prims=prims,
        frame=frame,
        samples=pts,
        tangents=tan,
        weights=np.ones(len(pts)),
        sample_prim=idx,
        diag=diag,
        label=label,
        piece_ids=list(piece_ids or []),
        variant=variant,
        box=box,
    )


def extract_query(
    spec: QuerySpec, index: PrimIndex, cfg: Config, strokes: dict[int, float] | None = None
) -> Query:
    """Эталон из нормализованного чертежа (индекс по его кускам).

    - Кандидаты: куски, рамка которых пересекает рамку эталона (расширенную на
      ``query_bbox_pad`` диагонали).
    - Кусок берется, если его штрих (все куски с тем же ``stroke_id``) лежит в рамке по длине
      >= ``query_parent_inside``, и сам кусок лежит в рамке хотя бы наполовину по длине.
      Так стена или размерная линия, проходящая через рамку, отбрасывается.
    - TextPrim берется, если центр его рамки в рамке эталона.
    - Нормализация: центр рамки -> 0, диагональ рамки -> 1; ``frame`` =
      Similarity(s=diag, theta=0, tx=cx, ty=cy) переводит нормализованное в чертеж.
    - Сэмплы и касательные с шагом ``sample_step`` (в нормализованных единицах),
      веса = 1 (до ``discriminative``), ``piece_ids`` = id взятых кусков.
    - ValueError, если в рамке не нашлось ни одного куска и текста.
    """
    diag = bbox_diag(spec.bbox)
    if diag <= 0:
        raise ValueError(f"эталон {spec.legend_id}: пустая рамка {spec.bbox}")
    box = bbox_expand(spec.bbox, cfg.query_bbox_pad * diag)
    if strokes is None:
        strokes = stroke_lengths(index)
    cand = index.query_bbox(box)
    inside_len: dict[int, float] = {}
    fr: dict[int, float] = {}
    for k in cand:
        p = index.prims[int(k)]
        if isinstance(p, TextPrim):
            continue
        f = _inside_frac(p, box)
        fr[int(k)] = f
        sid = p.stroke_id if p.stroke_id >= 0 else p.id
        inside_len[sid] = inside_len.get(sid, 0.0) + f * prim_length(p)
    chosen: list[Primitive] = []
    for k, f in fr.items():
        p = index.prims[k]
        sid = p.stroke_id if p.stroke_id >= 0 else p.id
        total = strokes.get(sid, 0.0)
        if f >= 0.5 and total > 0 and inside_len[sid] / total >= cfg.query_parent_inside:
            chosen.append(p)
    chosen.extend(index.texts_in_bbox(spec.bbox))
    if not chosen:
        raise ValueError(f"эталон {spec.legend_id}: в рамке {spec.bbox} нет примитивов")
    chosen.sort(key=lambda p: p.id)
    cx, cy = bbox_center(spec.bbox)
    frame = Similarity(diag, 0.0, cx, cy)
    inv = frame.inverse()
    prims_n = [transform_prim(p, inv) for p in chosen]
    x0, y0, x1, y1 = spec.bbox
    nbox = ((x0 - cx) / diag, (y0 - cy) / diag, (x1 - cx) / diag, (y1 - cy) / diag)
    return make_query(
        spec.legend_id,
        prims_n,
        frame,
        diag,
        cfg,
        label=spec.label,
        piece_ids=[p.id for p in chosen],
        box=nbox,
    )
