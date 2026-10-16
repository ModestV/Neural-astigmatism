"""Растровый конвейер. Фаза 5, Opus."""

from __future__ import annotations

import dataclasses
import logging

from symspot.core.config import Config
from symspot.core.types import Drawing, Instance, Query, QuerySpec

log = logging.getLogger("symspot")


def run_raster(
    drawing: Drawing, specs: list[QuerySpec], cfg: Config, timer
) -> tuple[list[Instance], Drawing]:
    """Предобработка -> эталоны по скелету -> chamfer-гипотезы -> ICP/покрытие ->
    соревнование -> маски -> разрешение (по рамкам)."""
    from symspot.pipeline import dedup_verified, group_queries, mark_query_instances
    from symspot.query.discriminative import apply_discriminative_weights
    from symspot.raster.chamfer import chamfer_search
    from symspot.raster.mask import instance_mask
    from symspot.raster.preprocess import preprocess
    from symspot.raster.verify_raster import RasterIndex, extract_raster_query
    from symspot.resolve.global_resolve import resolve
    from symspot.search.voting import nms_hypotheses
    from symspot.verify.compete import compete, query_for
    from symspot.verify.coverage import verify_hypothesis

    if drawing.image is None:
        raise ValueError("растровый чертеж без изображения")
    with timer("preprocess"):
        prep = preprocess(drawing.image, cfg)
    with timer("index"):
        index = RasterIndex(prep)
    with timer("query"):
        queries: list[Query] = []
        variants: dict[str, int] = {}
        for s in specs:
            try:
                q = extract_raster_query(s, prep, cfg)
            except ValueError as e:
                log.warning("%s", e)
                continue
            q.variant = variants.get(s.legend_id, 0)
            variants[s.legend_id] = q.variant + 1
            queries.append(q)
        queries = apply_discriminative_weights(queries, cfg)
    found: list[Instance] = []
    for q in queries:
        with timer("chamfer"):
            hyps = nms_hypotheses(chamfer_search(q, prep, cfg), q.diag, cfg)
        with timer("verify"):
            got = [verify_hypothesis(q, h, index, cfg) for h in hyps]
        found += dedup_verified([i for i in got if i is not None])
    qmap = group_queries(queries)
    with timer("compete"):
        found = compete(found, qmap, index, cfg)
    with timer("mask"):
        found = [
            dataclasses.replace(i, mask_rle=instance_mask(i, query_for(i, qmap), prep, cfg))
            for i in found
        ]
    with timer("resolve"):
        found = resolve(mark_query_instances(found, specs), qmap, cfg)
    return found, drawing
