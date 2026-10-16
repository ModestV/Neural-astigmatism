"""3.4б Проверка гипотезы покрытием. Ядро (Opus)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.geometry_utils import (
    bbox_center,
    bbox_diag,
    normalize_text,
    prim_length,
    prims_bbox,
    transform_prim,
)
from symspot.core.types import BBox, Hypothesis, Instance, Query, Similarity, TextPrim
from symspot.normalize.index import PrimIndex
from symspot.verify.icp import (
    ICP_START_RADIUS,
    ICP_START_RADIUS_TEXT,
    icp_refine,
    match_points,
)

# Текст эталона ищется на чертеже в пределах этой доли диагонали его рамки от ожидаемого центра.
TEXT_CENTER_TOL = 1.0
# Штрих, выходящий за область экземпляра больше чем на эту долю диагонали, считается
# проходящим насквозь и не входит в лишнюю тушь.
PASS_THROUGH_MARGIN = 0.25


def stroke_bboxes(index: PrimIndex) -> np.ndarray:
    """Рамка штриха (все куски с тем же ``stroke_id``) для каждого примитива индекса
    (запоминается в индексе)."""
    sb = getattr(index, "_symspot_stroke_bbox", None)
    if sb is None or len(sb) != len(index.prims):
        sids = np.array(
            [p.stroke_id if p.stroke_id >= 0 else -(p.id + 1) for p in index.prims], dtype=np.int64
        )
        sb = np.array(index.bboxes, dtype=float).reshape(-1, 4).copy()
        if len(sids):
            uniq, inv = np.unique(sids, return_inverse=True)
            mn = np.full((len(uniq), 2), np.inf)
            mx = np.full((len(uniq), 2), -np.inf)
            np.minimum.at(mn, inv, sb[:, :2])
            np.maximum.at(mx, inv, sb[:, 2:])
            sb = np.concatenate([mn[inv], mx[inv]], axis=1)
        index._symspot_stroke_bbox = sb  # type: ignore[attr-defined]
    return sb


def sample_half_width(index: PrimIndex) -> np.ndarray:
    """Половина толщины линии для каждого сэмпла чертежа (запоминается в индексе)."""
    hw = getattr(index, "_symspot_half_width", None)
    if hw is None or len(hw) != len(index.samples):
        widths = np.array(
            [getattr(getattr(p, "style", None), "stroke_width", 0.0) for p in index.prims],
            dtype=float,
        )
        hw = 0.5 * widths[index.sample_prim] if len(index.samples) else np.empty(0)
        index._symspot_half_width = hw  # type: ignore[attr-defined]
    return hw


def query_extent(query: Query) -> BBox:
    """Плотная рамка геометрии и текстов эталона в нормализованных координатах."""
    b = prims_bbox(query.prims)
    if b is None:
        if len(query.samples):
            mn, mx = query.samples.min(0), query.samples.max(0)
            return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))
        return (-0.5, -0.5, 0.5, 0.5)
    return b


def query_length(query: Query) -> float:
    """Длина линий эталона; для растрового эталона (без примитивов) — по сэмплам."""
    L = float(sum(prim_length(p) for p in query.prims if not isinstance(p, TextPrim)))
    if L == 0.0 and len(query.samples) > 1:
        d, _ = cKDTree(query.samples).query(query.samples, k=2)
        L = float(len(query.samples) * np.median(d[:, 1]))
    return L


@dataclass
class CoverageReport:
    covered: np.ndarray  # (N,) bool по сэмплам эталона
    match_idx: np.ndarray  # (N,) индекс сэмпла чертежа или -1
    coverage_w: float
    coverage: float
    prim_coverage: list[float]
    text_ids: list[int]
    texts_ok: bool
    extra_ratio: float


def _tol_and_cos(T: Similarity, cfg: Config) -> tuple[float, float]:
    return cfg.cover_tol * T.s, math.cos(math.radians(cfg.icp_angle_deg))


def coverage_report(query: Query, T: Similarity, index: PrimIndex, cfg: Config) -> CoverageReport:
    """Покрытие эталона, тексты и лишняя тушь для фиксированного T."""
    tol, cos_min = _tol_and_cos(T, cfg)
    hw = sample_half_width(index)
    P = T.apply(query.samples) if len(query.samples) else np.empty((0, 2))
    Pt = T.apply_vec(query.tangents) if len(query.samples) else np.empty((0, 2))
    max_hw = float(hw.max()) if len(hw) else 0.0
    d, idx = match_points(P, Pt, index.samples, index.tangents, index.kdtree, tol + max_hw, cos_min)
    ok = idx >= 0
    if ok.any():
        ok[ok] = d[ok] <= tol + hw[idx[ok]]
    w = query.weights
    cov_w = float((w * ok).sum() / w.sum()) if len(w) and w.sum() > 0 else 1.0
    cov = float(ok.mean()) if len(ok) else 1.0
    prim_cov: list[float] = []
    for k, p in enumerate(query.prims):
        if isinstance(p, TextPrim):
            continue
        m = query.sample_prim == k
        if m.any():
            prim_cov.append(float(ok[m].mean()))

    # тексты
    text_ids: list[int] = []
    texts_ok = True
    for p in query.prims:
        if not isinstance(p, TextPrim):
            continue
        tp = transform_prim(p, T)
        assert isinstance(tp, TextPrim)
        size = max(bbox_diag(tp.bbox), 1e-9)
        cx, cy = bbox_center(tp.bbox)
        r = TEXT_CENTER_TOL * size
        found = None
        want = normalize_text(p.text)
        for t in index.texts_in_bbox((cx - r, cy - r, cx + r, cy + r)):
            if normalize_text(t.text) == want:
                found = t
                break
        if found is None:
            texts_ok = False
        else:
            text_ids.append(found.id)

    # лишняя тушь внутри плотной рамки эталона
    extra_ratio = 0.0
    qlen = query_length(query)
    ext = query_extent(query)
    pad = cfg.cover_tol
    ext = (ext[0] - pad, ext[1] - pad, ext[2] + pad, ext[3] + pad)
    if index.kdtree is not None and qlen > 0:
        corners = np.array([[ext[0], ext[1]], [ext[2], ext[1]], [ext[2], ext[3]], [ext[0], ext[3]]])
        rad = float(np.hypot(corners[:, 0], corners[:, 1]).max()) * T.s
        center = T.apply(np.zeros(2))
        near = index.kdtree.query_ball_point(center, r=rad)
        if near:
            near = np.asarray(near, dtype=np.int64)
            local = T.inverse().apply(index.samples[near])
            inside = (
                (local[:, 0] >= ext[0])
                & (local[:, 0] <= ext[2])
                & (local[:, 1] >= ext[1])
                & (local[:, 1] <= ext[3])
            )
            if inside.any() and len(index.prims):
                # линии, проходящие насквозь (их штрих выходит далеко за область), — не тушь
                # символа: стена или размерная линия через символ не штрафуется
                sb = stroke_bboxes(index)[index.sample_prim[near]]
                box_d = _transform_ext(ext, T)
                m = PASS_THROUGH_MARGIN * T.s
                contained = (
                    (sb[:, 0] >= box_d[0] - m)
                    & (sb[:, 1] >= box_d[1] - m)
                    & (sb[:, 2] <= box_d[2] + m)
                    & (sb[:, 3] <= box_d[3] + m)
                )
                inside &= contained
            if inside.any() and len(query.samples):
                qtree = cKDTree(query.samples)
                dd, _ = qtree.query(local[inside], k=1)
                lim = cfg.cover_tol + hw[near[inside]] / T.s
                n_extra = int((dd > lim).sum())
            else:
                n_extra = int(inside.sum())
            extra_len = n_extra * index.sample_step / T.s
            # чужие тексты внутри рамки
            box_d = _transform_ext(ext, T)
            for t in index.texts_in_bbox(box_d):
                if t.id not in text_ids:
                    extra_len += bbox_diag(t.bbox) / T.s
            extra_ratio = extra_len / qlen
    return CoverageReport(ok, idx, cov_w, cov, prim_cov, text_ids, texts_ok, extra_ratio)


def _transform_ext(ext: BBox, T: Similarity) -> BBox:
    c = T.apply(np.array([[ext[0], ext[1]], [ext[2], ext[1]], [ext[2], ext[3]], [ext[0], ext[3]]]))
    return (float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max()))


def instance_bbox(query: Query, T: Similarity) -> BBox:
    """Рамка экземпляра: плотная рамка трансформированного эталона + средний отступ рамки
    эталона от его геометрии (как у разметчика)."""
    tp = [transform_prim(p, T) for p in query.prims]
    b = prims_bbox(tp)
    if b is None:
        b = _transform_ext(query_extent(query), T)
    if query.box is not None:
        ext = query_extent(query)
        m = np.mean(
            [
                ext[0] - query.box[0],
                ext[1] - query.box[1],
                query.box[2] - ext[2],
                query.box[3] - ext[3],
            ]
        )
        m = max(0.0, float(m)) * T.s
    else:
        m = 0.0
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def score_report(rep: CoverageReport, cfg: Config) -> float:
    return rep.coverage_w - cfg.extra_ink_lambda * rep.extra_ratio


def verify_hypothesis(
    query: Query, hyp: Hypothesis, index: PrimIndex, cfg: Config
) -> Instance | None:
    """ICP + покрытие. Возвращает Instance (без piece_ids) или None при отказе.

    - Покрытие = взвешенная доля сэмплов эталона, у которых есть сэмпл чертежа ближе
      ``cover_tol`` * диагональ экземпляра + половина толщины линии, с согласованной
      касательной.
    - Отказ, если покрытие какого-то примитива эталона < ``min_prim_coverage``.
    - Текст эталона должен найтись на трансформированном месте с той же нормализованной
      строкой (при ``text_match``), иначе отказ.
    - score = weighted_coverage - ``extra_ink_lambda`` * extra_ink_ratio; отказ при
      score < ``accept_score``.
    """
    if len(query.samples) >= 3:
        r0 = ICP_START_RADIUS_TEXT if hyp.source == "text" else ICP_START_RADIUS
        T, residual = icp_refine(query, hyp.T, index, cfg, start_radius=r0)
        if not math.isfinite(residual):
            return None
    else:
        T, residual = hyp.T, 0.0
    rel = T.s / query.diag
    if not (cfg.scale_min <= rel <= cfg.scale_max):
        return None
    rep = coverage_report(query, T, index, cfg)
    if rep.prim_coverage and min(rep.prim_coverage) < cfg.min_prim_coverage:
        return None
    if cfg.text_match and not rep.texts_ok:
        return None
    score = score_report(rep, cfg)
    if score < cfg.accept_score:
        return None
    return Instance(
        legend_id=query.legend_id,
        T=T,
        score=float(score),
        coverage=rep.coverage_w,
        piece_ids=[],
        parent_ids=[],
        bbox=instance_bbox(query, T),
        variant=query.variant,
        source=hyp.source,
        prim_coverage=rep.prim_coverage,
        residual=float(residual),
    )
