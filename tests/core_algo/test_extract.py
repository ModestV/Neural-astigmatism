"""3.1 Извлечение эталона (PLAN.md, раздел 3.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from symspot.core.config import Config
from symspot.core.geometry_utils import bbox_center, bbox_diag
from symspot.core.types import QuerySpec, TextPrim
from symspot.query.extract import extract_query

from .helpers import (
    Scene,
    sim,
    sym_slash,
    sym_text_circle,
)


def _wall_scene():
    sc = Scene()
    t = sc.place("slash", sym_slash(), sim(1, 0, 100, 100))
    # длинная стена через рамку эталона и мимо ее центра
    sc.add_wall((0, 104), (400, 104))
    sc.build()
    return sc, t


def test_wall_through_frame_not_in_query_symbol_pieces_are_in():
    sc, t = _wall_scene()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    wall_pieces = set(sc.wall_pieces())
    sym_pieces = set(sc.pieces_of(t.parents))
    got = set(q.piece_ids)
    assert wall_pieces, "в сцене стена должна быть порезана на куски"
    assert not (got & wall_pieces), "куски стены попали в эталон"
    assert sym_pieces <= got, "куски символа (в том числе порезанные стеной) должны быть в эталоне"


def test_wall_through_frame_gives_same_query_geometry_as_without_wall():
    sc, t = _wall_scene()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    clean = Scene()
    t2 = clean.place("slash", sym_slash(), sim(1, 0, 100, 100))
    clean.build()
    q0 = extract_query(clean.spec(t2), clean.index, clean.cfg)
    # у эталона со стеной больше кусков (резы), но сэмплы тянутся по той же геометрии
    from scipy.spatial import cKDTree

    d, _ = cKDTree(q0.samples).query(q.samples)
    assert float(d.max()) < 0.02
    d2, _ = cKDTree(q.samples).query(q0.samples)
    assert float(d2.max()) < 0.02


def test_text_inside_frame_is_taken_and_text_outside_is_not():
    sc = Scene()
    t = sc.place("co", sym_text_circle("CO"), sim(1, 0, 100, 100))
    sc.add_text("OUT", (140, 95, 160, 105))  # центр вне рамки
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    assert q.texts == ["CO"]
    assert any(isinstance(p, TextPrim) for p in q.prims)


def test_normalization_center_diag_and_frame():
    sc, t = _wall_scene()
    box = t.box
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    diag = bbox_diag(box)
    cx, cy = bbox_center(box)
    assert q.diag == pytest.approx(diag)
    assert q.legend_id == "slash"
    assert q.variant == 0
    # рамка в нормализованных координатах: центр в нуле, диагональ 1
    assert q.box is not None
    assert bbox_center(q.box) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert bbox_diag(q.box) == pytest.approx(1.0, abs=1e-9)
    # frame переводит нормализованное обратно в чертеж
    np.testing.assert_allclose(q.frame.apply(np.array([0.0, 0.0])), [cx, cy], atol=1e-9)
    assert q.frame.s == pytest.approx(diag)
    assert q.frame.theta == pytest.approx(0.0)
    assert not q.frame.mirror
    back = q.frame.apply(np.array([[q.box[2], 0.0]]))
    assert back[0, 0] == pytest.approx(cx + q.box[2] * diag)


def test_frame_maps_query_samples_onto_drawing_geometry():
    sc, t = _wall_scene()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    pts = q.frame.apply(q.samples)
    dist, ind = sc.index.nearest_sample(pts, 0.05 * q.diag)
    assert np.all(ind >= 0), "образ каждого сэмпла эталона должен лежать на линиях чертежа"
    assert float(np.max(dist)) < 0.01 * q.diag


def test_samples_tangents_weights_are_consistent():
    sc, t = _wall_scene()
    cfg = sc.cfg
    q = extract_query(sc.spec(t), sc.index, cfg)
    n = len(q.samples)
    assert n > 20
    assert q.samples.shape == (n, 2)
    assert q.tangents.shape == (n, 2)
    assert q.weights.shape == (n,)
    assert q.sample_prim.shape == (n,)
    np.testing.assert_allclose(np.linalg.norm(q.tangents, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(q.weights, 1.0)
    assert q.sample_prim.min() >= 0 and q.sample_prim.max() < len(q.prims)
    assert q.complexity == n
    # шаг сэмплов ~ sample_step в нормализованных единицах: суммарная длина / шаг
    total_len = math.pi * 2 * 10 / q.diag + 2 * 8.5 * math.sqrt(2) / q.diag
    assert n == pytest.approx(total_len / cfg.sample_step, rel=0.25)


def test_normalized_prims_are_in_unit_system():
    sc, t = _wall_scene()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    assert q.samples.min() >= -0.6 and q.samples.max() <= 0.6
    assert np.abs(q.samples).max() > 0.2


def test_empty_frame_raises_value_error():
    sc, _ = _wall_scene()
    with pytest.raises(ValueError):
        extract_query(QuerySpec("void", (400, 400, 440, 440)), sc.index, sc.cfg)


def test_frame_on_empty_drawing_raises_value_error():
    from symspot.normalize.index import PrimIndex

    idx = PrimIndex([], 1.0)
    with pytest.raises(ValueError):
        extract_query(QuerySpec("void", (0, 0, 10, 10)), idx, Config())


def test_degenerate_zero_size_frame_on_blank_area_raises_value_error():
    sc, _ = _wall_scene()
    with pytest.raises(ValueError):
        extract_query(QuerySpec("point", (500, 500, 500, 500)), sc.index, sc.cfg)


def test_query_is_independent_of_other_far_symbols():
    sc = Scene()
    t = sc.place("slash", sym_slash(), sim(1, 0, 100, 100))
    sc.place("slash", sym_slash(), sim(1.1, 20, 400, 400))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    assert set(q.piece_ids) == set(sc.pieces_of(t.parents))
