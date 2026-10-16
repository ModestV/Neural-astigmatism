"""3.8 Расширение запроса уверенными экземплярами. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.geometry_utils import transform_prim
from symspot.core.types import Instance, Query
from symspot.normalize.index import PrimIndex
from symspot.query.extract import make_query


def expand_queries(
    instances: list[Instance], queries: dict[str, list[Query]], index: PrimIndex, cfg: Config
) -> list[Query]:
    """Экземпляры со score >= ``expand_score`` и остатком ICP > ``expand_residual`` становятся
    дополнительными эталонами своего класса (``variant`` > 0): эталон строится из кусков
    экземпляра, переведенных в нормализованную систему через T^-1. Не больше одного нового
    варианта на класс (самый уверенный).
    """
    best: dict[str, Instance] = {}
    for inst in instances:
        if inst.is_query or not inst.piece_ids:
            continue
        if inst.score < cfg.expand_score or inst.residual <= cfg.expand_residual:
            continue
        cur = best.get(inst.legend_id)
        if cur is None or inst.score > cur.score:
            best[inst.legend_id] = inst
    out: list[Query] = []
    for lid, inst in sorted(best.items()):
        base = queries[lid][0]
        inv = inst.T.inverse()
        prims = []
        for pid in inst.piece_ids:
            k = index.id_to_idx.get(pid)
            if k is not None:
                prims.append(transform_prim(index.prims[k], inv))
        if not prims:
            continue
        variant = max(q.variant for q in queries[lid]) + 1
        q = make_query(
            lid,
            prims,
            inst.T,
            base.diag,
            cfg,
            label=base.label,
            piece_ids=list(inst.piece_ids),
            variant=variant,
            box=base.box,
        )
        # веса отличающих частей переносятся по ближайшему сэмплу исходного эталона
        if len(base.samples) and len(q.samples):
            from scipy.spatial import cKDTree

            _, j = cKDTree(base.samples).query(q.samples, k=1)
            q.weights = base.weights[j].copy()
        out.append(q)
    return out
