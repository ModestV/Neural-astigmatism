"""3.5б Соревнование экземпляров разных классов. Ядро (Opus)."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.geometry_utils import bbox_area, bbox_diag
from symspot.core.types import Instance, Query
from symspot.normalize.index import PrimIndex
from symspot.verify.coverage import CoverageReport, coverage_report


def query_for(instance: Instance, queries: dict[str, list[Query]]) -> Query:
    """Эталон (с вариантом) экземпляра."""
    qs = queries[instance.legend_id]
    for q in qs:
        if q.variant == instance.variant:
            return q
    return qs[0]


def overlap_min(a, b) -> float:
    """Пересечение рамок, деленное на площадь меньшей. В отличие от IoU ловит вложенные
    символы (duplex внутри рамки GFCI: IoU ~0.4, но это одно место на чертеже)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min(bbox_area(a), bbox_area(b))
    return ix * iy / m if m > 0 else 0.0


def _evidence(
    inst: Instance, q: Query, rep: CoverageReport, index: PrimIndex, cfg: Config
) -> tuple[np.ndarray, np.ndarray, float, dict[int, float]]:
    """Тушь экземпляра на чертеже: покрытые точки, все точки модели, длина шага, тексты."""
    pts_all = inst.T.apply(q.samples) if len(q.samples) else np.empty((0, 2))
    covered = pts_all[rep.covered] if len(pts_all) else pts_all
    step = cfg.sample_step * inst.T.s
    texts = {}
    for tid in rep.text_ids:
        k = index.id_to_idx.get(tid)
        if k is not None:
            texts[tid] = bbox_diag(index.prims[k].bbox)  # type: ignore[union-attr]
    return covered, pts_all, step, texts


def _explain(
    a: tuple[np.ndarray, np.ndarray, float, dict[int, float]],
    b: tuple[np.ndarray, np.ndarray, float, dict[int, float]],
    tol: float,
    w: float,
) -> tuple[float, float]:
    """Доли объединенной туши, объясненные моделями a и b (отличающие части весят ``w``)."""
    cov_a, all_a, step_a, txt_a = a
    cov_b, all_b, step_b, txt_b = b

    def near(points: np.ndarray, model: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        if len(model) == 0:
            return np.zeros(len(points), dtype=bool)
        d, _ = cKDTree(model).query(points, k=1)
        return d <= tol

    a_in_b = near(cov_a, all_b)
    b_in_a = near(cov_b, all_a)
    shared = a_in_b.sum() * step_a
    a_only = (~a_in_b).sum() * step_a
    b_only = (~b_in_a).sum() * step_b
    for tid, size in txt_a.items():
        if tid in txt_b:
            shared += size
        else:
            a_only += size
    for tid, size in txt_b.items():
        if tid not in txt_a:
            b_only += size
    total = shared + w * (a_only + b_only)
    if total <= 0:
        return 0.0, 0.0
    return (shared + w * a_only) / total, (shared + w * b_only) / total


def compete(
    instances: list[Instance],
    queries: dict[str, list[Query]],
    index: PrimIndex,
    cfg: Config,
) -> list[Instance]:
    """Экземпляры разных классов с перекрытием рамок (пересечение / меньшая площадь) >
    ``compete_iou`` сравниваются: побеждает
    больший score с отрывом >= ``compete_margin`` (проигравший удаляется), иначе оба
    остаются со ``status="ambiguous"`` и взаимными ``rivals``. Score в соревновании учитывает
    отличающие части обоих эталонов (присутствие отличающих частей соперника на чертеже —
    свидетельство против).

    Реализация: сначала сравнивается доля объединенной туши обоих экземпляров, которую
    объясняет каждая модель (отличающие части весят ``discriminative_weight``); при
    отрыве < ``compete_margin`` — обычный score; при равенстве и там — ambiguous.
    """
    n = len(instances)
    if n < 2:
        return list(instances)
    boxes = np.array([i.bbox for i in instances], dtype=float)
    alive = np.ones(n, dtype=bool)
    status = [i.status for i in instances]
    rivals: list[set[str]] = [set(i.rivals) for i in instances]
    cache: dict[int, tuple] = {}

    def ev(k: int):
        if k not in cache:
            inst = instances[k]
            q = query_for(inst, queries)
            rep = coverage_report(q, inst.T, index, cfg)
            cache[k] = _evidence(inst, q, rep, index, cfg)
        return cache[k]

    # кандидаты пар через пересечение рамок
    pairs: list[tuple[float, int, int]] = []
    order = np.argsort(boxes[:, 0])
    for oi, a in enumerate(order):
        for b in order[oi + 1 :]:
            if boxes[b, 0] > boxes[a, 2]:
                break
            if instances[a].legend_id == instances[b].legend_id:
                continue
            iou = overlap_min(instances[a].bbox, instances[b].bbox)
            if iou > cfg.compete_iou:
                pairs.append((iou, int(a), int(b)))
    pairs.sort(key=lambda t: -t[0])
    for _, a, b in pairs:
        if not (alive[a] and alive[b]):
            continue
        ia, ib = instances[a], instances[b]
        tol = cfg.cover_tol * max(ia.T.s, ib.T.s)
        ea, eb = _explain(ev(a), ev(b), tol, cfg.discriminative_weight)
        if abs(ea - eb) >= cfg.compete_margin:
            loser = b if ea > eb else a
        elif abs(ia.score - ib.score) >= cfg.compete_margin:
            loser = b if ia.score > ib.score else a
        else:
            status[a] = status[b] = "ambiguous"
            rivals[a].add(ib.legend_id)
            rivals[b].add(ia.legend_id)
            continue
        alive[loser] = False
    out = []
    for k in range(n):
        if alive[k]:
            out.append(
                dataclasses.replace(instances[k], status=status[k], rivals=sorted(rivals[k]))
            )
    return out
