"""3.5а Отличающие части похожих эталонов. Ядро (Opus)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.types import Query, Similarity
from symspot.verify.icp import align_point_sets, match_points

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


def directed_coverage(
    src: np.ndarray,
    src_t: np.ndarray,
    dst: np.ndarray,
    dst_t: np.ndarray,
    tol: float,
    cos_min: float,
) -> np.ndarray:
    """Маска сэмплов ``src``, у которых есть сэмпл ``dst`` ближе ``tol`` с согласованной
    касательной."""
    if len(src) == 0 or len(dst) == 0:
        return np.zeros(len(src), dtype=bool)
    tree = cKDTree(dst)
    _, idx = match_points(src, src_t, dst, dst_t, tree, tol, cos_min)
    return idx >= 0


def best_alignment(a: Query, b: Query, cfg: Config) -> tuple[Similarity, float]:
    """Лучшее совмещение сэмплов ``b`` с сэмплами ``a`` в нормализованных координатах ``a``.

    Возвращает (T: нормализованные b -> нормализованные a, сходство) — сходство = большее из
    двух направленных покрытий (доля сэмплов одного эталона, покрытых другим с допуском
    ``cover_tol`` и согласованной касательной). Большее, потому что один эталон может быть
    частью другого (⊘ внутри X, duplex внутри GFCI). Грубый проход ICP по начальным
    поворотам и отражению, выбор по сходству, затем полное уточнение лучшего.
    """
    if len(a.samples) < 3 or len(b.samples) < 3:
        return Similarity(), 0.0
    # старт по центроидам и разбросу сэмплов: не зависит от рамок (у GFCI рамка с текстом)
    ca, cb = a.samples.mean(0), b.samples.mean(0)
    ra = float(np.sqrt(((a.samples - ca) ** 2).sum(1).mean()))
    rb = float(np.sqrt(((b.samples - cb) ** 2).sum(1).mean()))
    s0 = ra / max(rb, 1e-12)
    mirrors = (False, True) if cfg.allow_mirror else (False,)
    r0 = INIT_RADIUS
    r1 = cfg.cover_tol
    cos_min = math.cos(math.radians(cfg.icp_angle_deg))
    coarse = cfg.model_copy(update={"icp_iters": COARSE_ITERS})

    def similarity(T: Similarity) -> float:
        tb = T.apply(b.samples)
        tbt = T.apply_vec(b.tangents)
        cov_ba = directed_coverage(tb, tbt, a.samples, a.tangents, r1, cos_min).mean()
        cov_ab = directed_coverage(a.samples, a.tangents, tb, tbt, r1, cos_min).mean()
        return float(max(cov_ab, cov_ba))

    cands: list[tuple[float, Similarity]] = []
    for mirror in mirrors:
        for deg in INIT_ROTATIONS_DEG:
            R0 = Similarity(s0, math.radians(deg), 0.0, 0.0, mirror)
            t0 = ca - R0.apply(cb)
            T0 = Similarity(s0, R0.theta, float(t0[0]), float(t0[1]), mirror)
            T, ch = align_point_sets(
                b.samples, b.tangents, a.samples, a.tangents, T0, coarse, r0, r1
            )
            if math.isfinite(ch):
                cands.append((similarity(T), T))
    if not cands:
        return Similarity(), 0.0
    sim_c, Tc = max(cands, key=lambda c: c[0])
    T, _ = align_point_sets(b.samples, b.tangents, a.samples, a.tangents, Tc, cfg, r1 * 3, r1)
    sim = similarity(T)
    if sim < sim_c:
        return Tc, sim_c
    return T, sim


def apply_discriminative_weights(queries: list[Query], cfg: Config) -> list[Query]:
    """Для каждой пары эталонов схожего размера (отношение диагоналей <
    ``similar_size_ratio``) совмещает их (ICP с перебором поворотов/отражения); если они
    похожи (один покрывает другой на >= ``similar_cover``), сэмплам одного, не покрытым
    другим, вес умножается на ``discriminative_weight``. Тексты эталона, которых нет в
    похожем эталоне, учитываются в соревновании (``verify/compete.py``). Возвращает новые
    Query (исходные не меняются).
    """
    cos_min = math.cos(math.radians(cfg.icp_angle_deg))
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
            T, sim = best_alignment(a, b, cfg)
            if sim < cfg.similar_cover:
                continue
            covered = directed_coverage(
                a.samples,
                a.tangents,
                T.apply(b.samples),
                T.apply_vec(b.tangents),
                cfg.cover_tol,
                cos_min,
            )
            disc = ~covered
            weights[i][disc] = np.maximum(weights[i][disc], cfg.discriminative_weight)
    return [dataclasses.replace(q, weights=w) for q, w in zip(queries, weights, strict=True)]
