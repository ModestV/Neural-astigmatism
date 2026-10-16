"""3.8 Расширение запроса."""

from __future__ import annotations

import numpy as np
import pytest

from symspot.core.types import Hypothesis, Instance
from symspot.expand.query_expansion import expand_queries
from symspot.query.extract import extract_query
from symspot.verify.coverage import verify_hypothesis

from .helpers import Scene, sim, sym_slash, sym_tri, true_T


def _setup():
    sc = Scene()
    t = sc.place("slash", sym_slash(), sim(1.0, 0, 100, 100))
    c1 = sc.place("slash", sym_slash(), sim(1.1, 20, 400, 300))
    c2 = sc.place("slash", sym_slash(), sim(0.9, -50, 400, 480))
    t2 = sc.place("tri", sym_tri(), sim(1.0, 0, 100, 250))
    c3 = sc.place("tri", sym_tri(), sim(1.0, 70, 250, 480, True))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    q2 = extract_query(sc.spec(t2), sc.index, sc.cfg)
    return sc, t, c1, c2, t2, c3, q, q2


def _inst(sc, q, t, c, score, residual, legend=None) -> Instance:
    T = true_T(q, t, c)
    pieces = sc.pieces_of(c.parents)
    parents = sorted(set(c.parents))
    return Instance(
        legend_id=legend or q.legend_id,
        T=T,
        score=score,
        coverage=score,
        piece_ids=pieces,
        parent_ids=parents,
        bbox=c.box,
        residual=residual,
    )


def test_confident_deviating_instance_becomes_new_query():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    inst = _inst(sc, q, t, c1, score=0.95, residual=0.03)
    out = expand_queries([inst], {"slash": [q]}, sc.index, sc.cfg)
    assert len(out) == 1
    nq = out[0]
    assert nq.legend_id == "slash"
    assert nq.variant > 0
    assert len(nq.samples) > 20
    assert nq.samples.shape == nq.tangents.shape == (len(nq.samples), 2)
    assert nq.weights.shape == (len(nq.samples),)
    assert set(nq.piece_ids) == set(inst.piece_ids)


def test_new_query_geometry_matches_instance_geometry_in_drawing():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    inst = _inst(sc, q, t, c1, score=0.95, residual=0.03)
    nq = expand_queries([inst], {"slash": [q]}, sc.index, sc.cfg)[0]
    pts = nq.frame.apply(nq.samples)
    dist, ind = sc.index.nearest_sample(pts, 0.05 * nq.diag)
    assert np.all(ind >= 0)
    assert float(dist.max()) < 0.03 * nq.diag
    # нормализация та же: рамка эталона укладывается в единичную диагональ
    assert np.abs(nq.samples).max() < 0.75


def test_low_score_instance_does_not_expand():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    inst = _inst(sc, q, t, c1, score=0.85, residual=0.05)
    assert expand_queries([inst], {"slash": [q]}, sc.index, sc.cfg) == []


def test_instance_close_to_query_does_not_expand():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    inst = _inst(sc, q, t, c1, score=0.99, residual=0.002)
    assert expand_queries([inst], {"slash": [q]}, sc.index, sc.cfg) == []


def test_score_exactly_at_threshold_expands():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    inst = _inst(sc, q, t, c1, score=sc.cfg.expand_score, residual=0.03)
    assert len(expand_queries([inst], {"slash": [q]}, sc.index, sc.cfg)) == 1


def test_empty_instances():
    sc, *_, q, q2 = _setup()
    assert expand_queries([], {"slash": [q]}, sc.index, sc.cfg) == []


def test_at_most_one_new_variant_per_class_and_it_is_the_most_confident():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    weak = _inst(sc, q, t, c1, score=0.92, residual=0.03)
    strong = _inst(sc, q, t, c2, score=0.98, residual=0.03)
    out = expand_queries([weak, strong], {"slash": [q]}, sc.index, sc.cfg)
    assert len(out) == 1
    assert set(out[0].piece_ids) == set(strong.piece_ids)


def test_each_class_expands_separately():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    a = _inst(sc, q, t, c1, score=0.95, residual=0.03)
    b = _inst(sc, q2, t2, c3, score=0.95, residual=0.03)
    out = expand_queries([a, b], {"slash": [q], "tri": [q2]}, sc.index, sc.cfg)
    assert sorted(x.legend_id for x in out) == ["slash", "tri"]
    assert all(x.variant > 0 for x in out)


def test_mixed_qualifying_and_not_only_qualifying_expands():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    a = _inst(sc, q, t, c1, score=0.95, residual=0.03)
    b = _inst(sc, q2, t2, c3, score=0.5, residual=0.5)
    out = expand_queries([a, b], {"slash": [q], "tri": [q2]}, sc.index, sc.cfg)
    assert [x.legend_id for x in out] == ["slash"]


def test_expanded_query_can_be_used_for_verification():
    sc, t, c1, c2, t2, c3, q, q2 = _setup()
    inst = _inst(sc, q, t, c1, score=0.95, residual=0.03)
    nq = expand_queries([inst], {"slash": [q]}, sc.index, sc.cfg)[0]
    h = Hypothesis("slash", nq.frame, 5.0, "expand", nq.variant)
    res = verify_hypothesis(nq, h, sc.index, sc.cfg)
    assert res is not None
    assert res.score == pytest.approx(1.0, abs=0.05)
