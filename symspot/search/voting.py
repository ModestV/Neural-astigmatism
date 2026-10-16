"""3.3 Голосование за преобразование. Ядро (Opus)."""

from __future__ import annotations

import math

import numpy as np

from symspot.core.config import Config
from symspot.core.geometry_utils import normalize_text, transform_prim
from symspot.core.types import Hypothesis, Query, Similarity, TextPrim
from symspot.normalize.index import PrimIndex
from symspot.query.descriptors import (
    CONCENTRIC_FRAC,
    KIND_SEGMENT,
    KIND_TEXT,
    compute_anchors,
    pair_keys,
    strokes_from_pieces,
)
from symspot.search.pairs import PairTable

MIRROR_Y = Similarity(1.0, 0.0, 0.0, 0.0, True)


def _query_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    m = ii != jj
    return ii[m], jj[m]


def _pair_transforms(
    qa, da, qi: np.ndarray, qj: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Подобия (без отражения) из совпавших пар: эталон (qi, qj) -> чертеж (a, b).

    Возвращает (log_s, theta, tx, ty, src) где src — индекс совпадения (для концентричных
    пар с отрезком дается два варианта поворота, src повторяется).
    """
    pi, pj = qa.pos[qi], qa.pos[qj]
    pa, pb = da.pos[a], da.pos[b]
    va = pj - pi
    vb = pb - pa
    la = np.hypot(va[:, 0], va[:, 1])
    lb = np.hypot(vb[:, 0], vb[:, 1])
    big_q = np.maximum(qa.size[qi], qa.size[qj])
    conc = la < CONCENTRIC_FRAC * np.maximum(big_q, 1e-12)
    idx = np.arange(len(qi))

    out_ls, out_th, out_src, out_c_q, out_c_d = [], [], [], [], []
    # различные якоря: по двум точкам
    nd = ~conc & (la > 1e-12) & (lb > 1e-12)
    if nd.any():
        s = lb[nd] / la[nd]
        th = np.arctan2(vb[nd, 1], vb[nd, 0]) - np.arctan2(va[nd, 1], va[nd, 0])
        out_ls.append(np.log(s))
        out_th.append(th)
        out_src.append(idx[nd])
        out_c_q.append(pi[nd])
        out_c_d.append(pa[nd])
    # концентричные: центр, отношение размеров, направление
    if conc.any():
        ci = idx[conc]
        qi_c, qj_c, a_c, b_c = qi[ci], qj[ci], a[ci], b[ci]
        ls = 0.5 * (
            np.log(np.maximum(da.size[a_c], 1e-12) / np.maximum(qa.size[qi_c], 1e-12))
            + np.log(np.maximum(da.size[b_c], 1e-12) / np.maximum(qa.size[qj_c], 1e-12))
        )
        # опорное направление: у первого примитива пары, если есть, иначе у второго
        use_i = ~np.isnan(qa.dir[qi_c]) & ~np.isnan(da.dir[a_c])
        qdir = np.where(use_i, qa.dir[qi_c], qa.dir[qj_c])
        ddir = np.where(use_i, da.dir[a_c], da.dir[b_c])
        qkind = np.where(use_i, qa.kind[qi_c], qa.kind[qj_c])
        ok = ~np.isnan(qdir) & ~np.isnan(ddir)
        th = ddir - qdir
        center_q = 0.5 * (qa.pos[qi_c] + qa.pos[qj_c])
        center_d = 0.5 * (da.pos[a_c] + da.pos[b_c])
        out_ls.append(ls[ok])
        out_th.append(th[ok])
        out_src.append(ci[ok])
        out_c_q.append(center_q[ok])
        out_c_d.append(center_d[ok])
        seg = ok & (qkind == KIND_SEGMENT)
        if seg.any():
            out_ls.append(ls[seg])
            out_th.append(th[seg] + math.pi)
            out_src.append(ci[seg])
            out_c_q.append(center_q[seg])
            out_c_d.append(center_d[seg])
    if not out_ls:
        e = np.empty(0)
        return e, e, e, e, np.empty(0, dtype=np.int64)
    ls = np.concatenate(out_ls)
    th = np.concatenate(out_th)
    src = np.concatenate(out_src)
    cq = np.concatenate(out_c_q)
    cd = np.concatenate(out_c_d)
    s = np.exp(ls)
    c, si = np.cos(th), np.sin(th)
    tx = cd[:, 0] - s * (c * cq[:, 0] - si * cq[:, 1])
    ty = cd[:, 1] - s * (si * cq[:, 0] + c * cq[:, 1])
    return ls, th, tx, ty, src


def _vote_one(
    query: Query, prims_n: list, mirror: bool, table: PairTable, index: PrimIndex, cfg: Config
) -> list[Hypothesis]:
    """Голосование для эталона (уже отраженного, если ``mirror``)."""
    qa = compute_anchors(strokes_from_pieces(prims_n))
    n = len(qa)
    if n < 2 or len(table) == 0:
        return []
    qi_all, qj_all = _query_pairs(n)
    qkeys, qrows = pair_keys(qa, qi_all, qj_all, cfg, spread=True)
    if len(qkeys) == 0:
        return []
    k, rows = table.lookup(qkeys)
    if len(k) == 0:
        return []
    qpair = qrows[k]
    # одна пара эталона может совпасть с одной строкой таблицы через несколько ключей
    uniq = np.unique(np.stack([qpair, rows], axis=1), axis=0)
    qpair, rows = uniq[:, 0], uniq[:, 1]
    qi, qj = qi_all[qpair], qj_all[qpair]
    ls, th, tx, ty, src = _pair_transforms(qa, table.anchors, qi, qj, table.a[rows], table.b[rows])
    if len(ls) == 0:
        return []
    # масштаб относительно эталона: T.s = rel * diag (нормализованная диагональ = 1)
    rel = ls - math.log(query.diag)
    keep = (rel >= math.log(cfg.scale_min)) & (rel <= math.log(cfg.scale_max))
    ls, th, tx, ty, src, rel = ls[keep], th[keep], tx[keep], ty[keep], src[keep], rel[keep]
    if len(ls) == 0:
        return []
    # неупорядоченная пара эталона — для счета различных пар
    upair = np.minimum(qi[src], qj[src]) * n + np.maximum(qi[src], qj[src])
    # сколько различных неупорядоченных пар эталона вообще имеют ключ
    valid_rows = np.unique(qrows)
    n_avail = len(
        np.unique(
            np.minimum(qi_all[valid_rows], qj_all[valid_rows]) * n
            + np.maximum(qi_all[valid_rows], qj_all[valid_rows])
        )
    )
    need = max(1, min(cfg.min_distinct_pairs, n_avail))

    shift = cfg.shift_bin * query.diag
    ang_w = math.radians(cfg.angle_bin_deg)
    n_ang = max(1, round(2 * math.pi / ang_w))
    th = np.mod(th, 2 * math.pi)
    coords = [tx / shift, ty / shift, th / ang_w, rel / cfg.scale_bin_log]
    # мягкое квантование: каждый голос в 2 ближайшие ячейки по каждой оси
    b0 = [np.floor(c - 0.5).astype(np.int64) for c in coords]
    cells = []
    for mask in range(16):
        cell = [b0[d] + ((mask >> d) & 1) for d in range(4)]
        cell[2] = np.mod(cell[2], n_ang)
        cells.append(np.stack(cell, axis=1))
    cell_all = np.concatenate(cells, axis=0)
    vid = np.tile(np.arange(len(ls)), 16)
    up = upair[vid]
    # ячейка -> число различных пар эталона
    cu, cinv = np.unique(cell_all, axis=0, return_inverse=True)
    cinv = cinv.ravel()
    cp = np.unique(np.stack([cinv, up], axis=1), axis=0)
    distinct = np.bincount(cp[:, 0], minlength=len(cu))
    raw = np.bincount(cinv, minlength=len(cu))
    good = np.flatnonzero(distinct >= need)
    if len(good) == 0:
        return []
    # оценка T в ячейке: средние по голосам ячейки
    sel = np.isin(cinv, good)
    gi = cinv[sel]
    gv = vid[sel]
    remap = -np.ones(len(cu), dtype=np.int64)
    remap[good] = np.arange(len(good))
    g = remap[gi]
    cnt = np.bincount(g, minlength=len(good)).astype(float)
    m_ls = np.bincount(g, weights=ls[gv], minlength=len(good)) / cnt
    m_tx = np.bincount(g, weights=tx[gv], minlength=len(good)) / cnt
    m_ty = np.bincount(g, weights=ty[gv], minlength=len(good)) / cnt
    m_c = np.bincount(g, weights=np.cos(th[gv]), minlength=len(good))
    m_s = np.bincount(g, weights=np.sin(th[gv]), minlength=len(good))
    m_th = np.arctan2(m_s, m_c)
    votes = distinct[good] + raw[good] / (1.0 + raw[good].max())
    if cfg.density_norm and index.kdtree is not None:
        centers = np.stack([m_tx, m_ty], axis=1)
        local = index.kdtree.query_ball_point(centers, r=0.5 * query.diag, return_length=True)
        expected = max(1.0, float(len(query.samples)))
        dens = np.maximum(1.0, np.asarray(local, dtype=float) / expected)
        votes = votes / np.sqrt(dens)
    hyps = []
    for t in range(len(good)):
        T = Similarity(float(math.exp(m_ls[t])), float(m_th[t]), float(m_tx[t]), float(m_ty[t]))
        if mirror:
            T = T.compose(MIRROR_Y)
        hyps.append(Hypothesis(query.legend_id, T, float(votes[t]), "vote", query.variant))
    return hyps


def text_hypotheses(query: Query, index: PrimIndex, cfg: Config) -> list[Hypothesis]:
    """Гипотезы от совпадающих строк текста (для эталонов с текстом)."""
    qtexts = [p for p in query.prims if isinstance(p, TextPrim)]
    if not qtexts:
        return []
    by_str: dict[str, list[TextPrim]] = {}
    for p in index.prims:
        if isinstance(p, TextPrim):
            by_str.setdefault(normalize_text(p.text), []).append(p)
    hyps: list[Hypothesis] = []
    for qt in qtexts:
        cands = by_str.get(normalize_text(qt.text), [])
        qw, qh = qt.bbox[2] - qt.bbox[0], qt.bbox[3] - qt.bbox[1]
        qc = np.array([(qt.bbox[0] + qt.bbox[2]) / 2, (qt.bbox[1] + qt.bbox[3]) / 2])
        for dt in cands:
            theta = dt.angle - qt.angle
            dw, dh = dt.bbox[2] - dt.bbox[0], dt.bbox[3] - dt.bbox[1]
            # масштаб по площади рамки, если текст близок к осям; иначе — номинальный
            axis_dev = abs(math.sin(2 * dt.angle))
            if axis_dev < 0.1 and qw * qh > 0:
                s = math.sqrt(dw * dh / (qw * qh))
            else:
                s = query.diag
            rel = s / query.diag
            if not (cfg.scale_min * 0.9 <= rel <= cfg.scale_max * 1.1):
                s = query.diag
            dc = np.array([(dt.bbox[0] + dt.bbox[2]) / 2, (dt.bbox[1] + dt.bbox[3]) / 2])
            T0 = Similarity(s, theta, 0.0, 0.0)
            p = T0.apply(qc)
            T = Similarity(s, theta, float(dc[0] - p[0]), float(dc[1] - p[1]))
            hyps.append(Hypothesis(query.legend_id, T, 1.0, "vote", query.variant))
    return hyps


def nms_hypotheses(hyps: list[Hypothesis], diag: float, cfg: Config) -> list[Hypothesis]:
    """NMS: близкие центр (< ``shift_bin`` диагонали), угол (< 2 ячеек) и то же зеркало."""
    if not hyps:
        return []
    order = sorted(range(len(hyps)), key=lambda i: -hyps[i].votes)
    hs = [hyps[i] for i in order]
    c = np.array([[h.T.tx, h.T.ty] for h in hs])
    th = np.array([h.T.theta for h in hs])
    mir = np.array([h.T.mirror for h in hs])
    keep: list[int] = []
    alive = np.ones(len(hs), dtype=bool)
    rad = cfg.shift_bin * diag
    ang = 2 * math.radians(cfg.angle_bin_deg)
    for i in range(len(hs)):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) >= cfg.max_hypotheses_per_query:
            break
        d = np.hypot(c[:, 0] - c[i, 0], c[:, 1] - c[i, 1])
        da = np.abs(np.angle(np.exp(1j * (th - th[i]))))
        sup = (d < rad) & (da < ang) & (mir == mir[i])
        alive &= ~sup
    return [hs[i] for i in keep]


def find_candidates(
    query: Query, table: PairTable, index: PrimIndex, cfg: Config
) -> list[Hypothesis]:
    """Гипотезы T (нормализованный эталон -> чертеж) для одного эталона.

    - Совпавший ключ пары -> подобие по двум якорям (для концентричных пар — по центру,
      отношению размеров и направлению); при ``allow_mirror`` то же для отраженного эталона.
    - Голос в аккумулятор: центр (шаг ``shift_bin`` диагонали эталона), угол
      (``angle_bin_deg``), log-масштаб (``scale_bin_log``), зеркало. Масштаб вне
      ``[scale_min, scale_max]`` отбрасывается.
    - Ячейка принимается, если голоса пришли от >= ``min_distinct_pairs`` разных пар эталона;
      вес голоса нормируется на локальную плотность кусков.
    - Для эталонов с текстом — дополнительные гипотезы от совпадающих строк текста.
    - NMS по центру, не больше ``max_hypotheses_per_query``, по убыванию голосов.
    """
    hyps = _vote_one(query, list(query.prims), False, table, index, cfg)
    has_text = any(isinstance(p, TextPrim) for p in query.prims)
    if cfg.allow_mirror and not has_text:
        mirrored = [transform_prim(p, MIRROR_Y) for p in query.prims]
        hyps += _vote_one(query, mirrored, True, table, index, cfg)
    hyps += text_hypotheses(query, index, cfg)
    return nms_hypotheses(hyps, query.diag, cfg)


__all__ = ["find_candidates", "nms_hypotheses", "text_hypotheses", "KIND_TEXT"]
