"""Векторный и растровый конвейер. Фаза 4, Opus."""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from symspot.core.config import Config
from symspot.core.geometry_utils import bbox_diag, bbox_iou
from symspot.core.types import Drawing, Hypothesis, Instance, Query, QuerySpec, Result

log = logging.getLogger("symspot")

# Экземпляр считается самим эталоном (легендой), если IoU его рамки с рамкой эталона выше.
QUERY_SELF_IOU = 0.5
# Дубли одного эталона после проверки: IoU рамок выше — оставляем лучший.
VERIFIED_DUP_IOU = 0.6


class Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - t0


def group_queries(queries: list[Query]) -> dict[str, list[Query]]:
    out: dict[str, list[Query]] = {}
    for q in queries:
        out.setdefault(q.legend_id, []).append(q)
    return out


def dedup_verified(instances: list[Instance]) -> list[Instance]:
    """Убрать дубли одного класса с IoU рамок > VERIFIED_DUP_IOU (оставить лучший score)."""
    out: list[Instance] = []
    for inst in sorted(instances, key=lambda i: -i.score):
        if any(
            o.legend_id == inst.legend_id and bbox_iou(o.bbox, inst.bbox) > VERIFIED_DUP_IOU
            for o in out
        ):
            continue
        out.append(inst)
    return out


def mark_query_instances(instances: list[Instance], specs: list[QuerySpec]) -> list[Instance]:
    for inst in instances:
        inst.is_query = any(
            s.legend_id == inst.legend_id and bbox_iou(s.bbox, inst.bbox) > QUERY_SELF_IOU
            for s in specs
        )
    return instances


def _search_vector(queries: list[Query], table, index, cfg: Config, timer: Timer) -> list[Instance]:
    from symspot.exact.hashing import find_exact
    from symspot.search.voting import find_candidates, nms_hypotheses
    from symspot.verify.coverage import verify_hypothesis

    found: list[Instance] = []
    for q in queries:
        hyps: list[Hypothesis] = []
        with timer("exact"):
            try:
                hyps += find_exact(q, index, cfg)
            except NotImplementedError:
                pass
        with timer("vote"):
            hyps += find_candidates(q, table, index, cfg)
        hyps = nms_hypotheses(hyps, q.diag, cfg)
        with timer("verify"):
            inst = [verify_hypothesis(q, h, index, cfg) for h in hyps]
        got = dedup_verified([i for i in inst if i is not None])
        log.debug("query %s v%d: %d hyps -> %d inst", q.legend_id, q.variant, len(hyps), len(got))
        found += got
    return found


def run_vector(
    drawing: Drawing, specs: list[QuerySpec], cfg: Config, timer: Timer
) -> tuple[list[Instance], Drawing]:
    from symspot.expand.query_expansion import expand_queries
    from symspot.normalize import normalize_drawing
    from symspot.normalize.index import PrimIndex
    from symspot.paint.assign import assign_pieces
    from symspot.query.discriminative import apply_discriminative_weights
    from symspot.query.extract import extract_query, stroke_lengths
    from symspot.resolve.global_resolve import resolve
    from symspot.search.pairs import PairTable
    from symspot.verify.compete import compete, query_for

    diags = [bbox_diag(s.bbox) for s in specs]
    ref_size = statistics.median(diags)
    with timer("normalize"):
        norm = normalize_drawing(drawing, ref_size, cfg)
    with timer("index"):
        index = PrimIndex(norm.primitives, cfg.sample_step * min(diags))
    with timer("query"):
        strokes = stroke_lengths(index)
        queries: list[Query] = []
        variants: dict[str, int] = {}
        for s in specs:
            try:
                q = extract_query(s, index, cfg, strokes)
            except ValueError as e:
                log.warning("%s", e)
                continue
            q.variant = variants.get(s.legend_id, 0)
            variants[s.legend_id] = q.variant + 1
            queries.append(q)
        queries = apply_discriminative_weights(queries, cfg)
    if not queries:
        return [], norm
    with timer("pairs"):
        radius = cfg.pair_radius_factor * max(diags) * cfg.scale_max
        table = PairTable.build(index, radius, cfg)
    found = _search_vector(queries, table, index, cfg, timer)
    qmap = group_queries(queries)

    def finish(cands: list[Instance]) -> list[Instance]:
        with timer("compete"):
            cands = compete(cands, qmap, index, cfg)
        with timer("assign"):
            cands = [assign_pieces(i, query_for(i, qmap), index, cfg) for i in cands]
        with timer("resolve"):
            return resolve(mark_query_instances(cands, specs), qmap, cfg)

    result = finish(found)
    if cfg.expand:
        with timer("expand"):
            extra_q = expand_queries(result, qmap, index, cfg)
        if extra_q:
            for q in extra_q:
                qmap.setdefault(q.legend_id, []).append(q)
            more = _search_vector(extra_q, table, index, cfg, timer)
            result = finish(found + more)
    return result, norm


def run(
    drawing_path: str | Path,
    queries_path: str | Path,
    cfg: Config,
    page: int = 0,
) -> tuple[Result, Drawing]:
    """Прочитать чертеж и эталоны, найти экземпляры. Возвращает (Result, нормализованный
    Drawing для рендеринга)."""
    from symspot.io import read_drawing
    from symspot.io.annotations import load_queries

    timer = Timer()
    with timer("read"):
        drawing = read_drawing(drawing_path, page=page)
        specs = load_queries(
            queries_path, drawing_size=(drawing.width, drawing.height), image_name=str(drawing_path)
        )
    if not specs:
        raise ValueError(f"{queries_path}: нет эталонов")
    if drawing.kind == "vector":
        instances, out_drawing = run_vector(drawing, specs, cfg, timer)
    else:
        from symspot.raster.pipeline import run_raster

        instances, out_drawing = run_raster(drawing, specs, cfg, timer)
    total = sum(timer.timings.values())
    timer.timings["total"] = total
    res = Result(
        drawing=str(drawing_path),
        instances=instances,
        timings=dict(timer.timings),
        config=cfg.model_dump(),
        queries=specs,
        kind=drawing.kind,
        width=drawing.width,
        height=drawing.height,
        page=page,
    )
    return res, out_drawing
