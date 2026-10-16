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
# Начальный радиус сопоставления (доля диагонали) и число итераций грубого прохода.
INIT_RADIUS = 0.15
COARSE_ITERS = 4
# Эталоны с отношением абсолютной длины линий больше этого не сравниваются (быстрый отсев).
MAX_LENGTH_RATIO = 2.5


def _abs_length(q: Query) -> float:
    from symspot.verify.coverage import query_length

    return query_length(q) * q.diag


def best_alignment(a: Query, b: Query, cfg: Config) -> tuple[Similarity, float]:
    """Лучшее совмещение сэмплов ``b`` с сэмплами ``a`` в нормализованных координатах ``a``.

    Возвращает (T: нормализованные b -> нормализованные a, симметричный chamfer в долях
    диагонали ``a``) — минимум из двух направленных chamfer (один эталон может быть частью
    другого). Грубый проход (COARSE_ITERS итераций ICP) по всем начальным поворотам, затем
    полное уточнение лучшего.
    """
    if len(a.samples) < 3 or len(b.samples) < 3:
        return Similarity(), math.inf
    s0 = b.diag / a.diag
    mirrors = (False, True) if cfg.allow_mirror else (False,)
    r0 = INIT_RADIUS
    r1 = cfg.cover_tol
    coarse = cfg.model_copy(update={"icp_iters": COARSE_ITERS})
    tree_a = cKDTree(a.samples)

    def score(T: Similarity) -> float:
        tb = T.apply(b.samples)
        d_ba, _ = tree_a.query(tb, k=1)
        d_ab, _ = cKDTree(tb).query(a.samples, k=1)
        return min(float(np.minimum(d_ab, r0).mean()), float(np.minimum(d_ba, r0).mean()))

    cands: list[tuple[float, Similarity]] = []
    for mirror in mirrors:
        for deg in INIT_ROTATIONS_DEG:
            T0 = Similarity(s0, math.radians(deg), 0.0, 0.0, mirror)
            T, ch = align_point_sets(
                b.samples, b.tangents, a.samples, a.tangents, T0, coarse, r0, r1
            )
            if math.isfinite(ch):
                cands.append((score(T), T))
    if not cands:
        return Similarity(), math.inf
    _, Tc = min(cands, key=lambda c: c[0])
    T, _ = align_point_sets(b.samples, b.tangents, a.samples, a.tangents, Tc, cfg, r1 * 3, r1)
    return T, score(T)


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
            la, lb = _abs_length(a), _abs_length(b)
            if max(la, lb) > MAX_LENGTH_RATIO * max(min(la, lb), 1e-12):
                continue
            T, ch = best_alignment(a, b, cfg)
            if ch >= cfg.similar_chamfer:
                continue
            tb = T.apply(b.samples)
            d, _ = cKDTree(tb).query(a.samples, k=1)
            disc = d > cfg.cover_tol
            weights[i][disc] = np.maximum(weights[i][disc], cfg.discriminative_weight)
    return [dataclasses.replace(q, weights=w) for q, w in zip(queries, weights, strict=True)]
