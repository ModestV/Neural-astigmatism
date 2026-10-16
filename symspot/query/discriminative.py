"""3.5а Отличающие части похожих эталонов. Ядро (Opus)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.types import Query, Similarity
from symspot.verify.icp import align_point_sets

# Начальные повороты при совмещении двух эталонов (градусы).
INIT_ROTATIONS_DEG = tuple(range(0, 360, 45))


def best_alignment(a: Query, b: Query, cfg: Config) -> tuple[Similarity, float]:
    """Лучшее совмещение сэмплов ``b`` с сэмплами ``a`` в нормализованных координатах ``a``.

    Возвращает (T: нормализованные b -> нормализованные a, симметричный chamfer в долях
    диагонали ``a``) — минимум из двух направленных chamfer (один эталон может быть частью
    другого).
    """
    if len(a.samples) < 3 or len(b.samples) < 3:
        return Similarity(), math.inf
    s0 = b.diag / a.diag
    best = (Similarity(), math.inf)
    mirrors = (False, True) if cfg.allow_mirror else (False,)
    r0 = 0.15
    r1 = cfg.cover_tol
    for mirror in mirrors:
        for deg in INIT_ROTATIONS_DEG:
            T0 = Similarity(s0, math.radians(deg), 0.0, 0.0, mirror)
            T, ch_ba = align_point_sets(
                b.samples, b.tangents, a.samples, a.tangents, T0, cfg, r0, r1
            )
            if not math.isfinite(ch_ba):
                continue
            # обратное направление: a -> T(b)
            Tb = T.apply(b.samples)
            tree = cKDTree(Tb)
            d, _ = tree.query(a.samples, k=1)
            ch_ab = float(np.minimum(d, r0).mean())
            ch = min(ch_ab, ch_ba)
            if ch < best[1]:
                best = (T, ch)
    return best


def apply_discriminative_weights(queries: list[Query], cfg: Config) -> list[Query]:
    """Для каждой пары эталонов схожего размера (отношение диагоналей <
    ``similar_size_ratio``) совмещает их (ICP с перебором поворотов/отражения); если они
    похожи (chamfer < ``similar_chamfer``), сэмплам одного, не покрытым другим, вес умножается
    на ``discriminative_weight``. Тексты эталона, которых нет в похожем эталоне, тоже
    считаются отличающими. Возвращает новые Query (исходные не меняются).
    """
    weights = [q.weights.copy() for q in queries]
    for i, a in enumerate(queries):
        for j, b in enumerate(queries):
            if i == j or a.legend_id == b.legend_id:
                continue
            ratio = max(a.diag, b.diag) / max(min(a.diag, b.diag), 1e-12)
            if ratio >= cfg.similar_size_ratio:
                continue
            T, ch = best_alignment(a, b, cfg)
            if ch >= cfg.similar_chamfer:
                continue
            tb = T.apply(b.samples)
            d, _ = cKDTree(tb).query(a.samples, k=1)
            disc = d > cfg.cover_tol
            weights[i][disc] = np.maximum(weights[i][disc], cfg.discriminative_weight)
    return [dataclasses.replace(q, weights=w) for q, w in zip(queries, weights, strict=True)]
