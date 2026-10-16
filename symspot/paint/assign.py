"""3.6 Назначение кусков экземпляру. Ядро (Opus)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial import cKDTree

from symspot.core.config import Config
from symspot.core.geometry_utils import bbox_expand
from symspot.core.types import Instance, Query, TextPrim
from symspot.normalize.index import PrimIndex
from symspot.verify.coverage import coverage_report, sample_half_width


def assign_pieces(instance: Instance, query: Query, index: PrimIndex, cfg: Config) -> Instance:
    """Заполнить ``piece_ids``/``parent_ids``: кусок принадлежит экземпляру, если
    >= ``assign_inside`` его сэмплов ближе допуска покрытия к трансформированному эталону
    и касательные согласованы. Тексты эталона, найденные на месте, тоже назначаются.
    """
    T = instance.T
    tol = cfg.cover_tol * T.s
    cos_min = math.cos(math.radians(cfg.icp_angle_deg))
    piece_ids: list[int] = []
    parent_ids: set[int] = set()
    if len(query.samples):
        P = T.apply(query.samples)
        Pt = T.apply_vec(query.tangents)
        tree = cKDTree(P)
        hw = sample_half_width(index)
        cand = index.query_bbox(bbox_expand(instance.bbox, tol))
        for k in cand:
            p = index.prims[int(k)]
            if isinstance(p, TextPrim):
                continue
            s0, s1 = int(index.sample_start[k]), int(index.sample_start[k + 1])
            if s1 <= s0:
                continue
            sp = index.samples[s0:s1]
            st = index.tangents[s0:s1]
            d, j = tree.query(sp, k=1, distance_upper_bound=tol + float(hw[s0:s1].max()))
            ok = np.isfinite(d)
            if ok.any():
                jj = np.where(ok, j, 0)
                cosv = np.abs((Pt[jj] * st).sum(1))
                ok &= cosv >= cos_min
                ok &= d <= tol + hw[s0:s1]
            if ok.mean() >= cfg.assign_inside:
                piece_ids.append(p.id)
                parent_ids.add(p.parent_id)
    if any(isinstance(p, TextPrim) for p in query.prims):
        rep = coverage_report(query, T, index, cfg)
        for tid in rep.text_ids:
            k = index.id_to_idx.get(tid)
            if k is not None:
                piece_ids.append(tid)
                parent_ids.add(index.prims[k].parent_id)
    return dataclasses.replace(
        instance, piece_ids=sorted(set(piece_ids)), parent_ids=sorted(parent_ids)
    )
