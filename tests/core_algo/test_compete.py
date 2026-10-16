"""3.5б Соревнование экземпляров разных классов."""

from __future__ import annotations

import dataclasses

from symspot.core.types import Instance
from symspot.query.discriminative import apply_discriminative_weights
from symspot.query.extract import extract_query
from symspot.verify.compete import compete
from symspot.verify.coverage import verify_hypothesis

from .helpers import (
    Scene,
    make_hyp,
    sim,
    sym_duplex,
    sym_gfci,
    sym_slash,
    sym_xcircle,
    true_T,
)


def _setup(items, tests):
    """items: [(legend_id, local_fn)] шаблоны; tests: [(local_fn, S)] места проверки."""
    sc = Scene()
    tpls = []
    for k, (name, fn) in enumerate(items):
        tpls.append(sc.place(name, fn(), sim(1.0, 0, 80 + 120 * k, 80)))
    places = [sc.place("test", fn(), S) for fn, S in tests]
    sc.build()
    qs = [extract_query(sc.spec(p), sc.index, sc.cfg) for p in tpls]
    qs = apply_discriminative_weights(qs, sc.cfg)
    queries = {}
    for q in qs:
        queries.setdefault(q.legend_id, []).append(q)
    return sc, tpls, qs, places, queries


def _instances_at(sc, tpls, qs, place):
    out = []
    for t, q in zip(tpls, qs, strict=True):
        T = true_T(q, t, place)
        inst = verify_hypothesis(q, make_hyp(q.legend_id, T), sc.index, sc.cfg)
        if inst is not None:
            out.append(inst)
    return out


def test_xcircle_beats_slash_on_xcircle_place():
    sc, tpls, qs, (pl,), queries = _setup(
        [("slash", sym_slash), ("x", sym_xcircle)], [(sym_xcircle, sim(1.0, 20, 400, 300))]
    )
    insts = _instances_at(sc, tpls, qs, pl)
    assert {i.legend_id for i in insts} == {"slash", "x"}, (
        "лишняя тушь (вторая диагональ) почти не штрафуется: ⊘ должен проходить проверку на X"
    )
    out = compete(insts, queries, sc.index, sc.cfg)
    assert [i.legend_id for i in out] == ["x"]
    assert out[0].status == "ok"


def test_slash_wins_on_slash_place_when_xcircle_fails_verification():
    sc, tpls, qs, (pl,), queries = _setup(
        [("slash", sym_slash), ("x", sym_xcircle)], [(sym_slash, sim(1.0, -35, 400, 300))]
    )
    insts = _instances_at(sc, tpls, qs, pl)
    assert [i.legend_id for i in insts] == ["slash"]
    out = compete(insts, queries, sc.index, sc.cfg)
    assert [i.legend_id for i in out] == ["slash"]


def test_gfci_beats_duplex_on_gfci_place():
    sc, tpls, qs, (pl,), queries = _setup(
        [("duplex", sym_duplex), ("gfci", sym_gfci)], [(sym_gfci, sim(1.0, 0, 400, 300))]
    )
    insts = _instances_at(sc, tpls, qs, pl)
    assert {i.legend_id for i in insts} == {"duplex", "gfci"}, (
        "duplex (подмножество GFCI) должен проходить проверку на месте GFCI"
    )
    out = compete(insts, queries, sc.index, sc.cfg)
    assert [i.legend_id for i in out] == ["gfci"]


def test_duplex_wins_on_duplex_place():
    sc, tpls, qs, (pl,), queries = _setup(
        [("duplex", sym_duplex), ("gfci", sym_gfci)], [(sym_duplex, sim(1.0, 0, 400, 300))]
    )
    insts = _instances_at(sc, tpls, qs, pl)
    assert [i.legend_id for i in insts] == ["duplex"], "GFCI без текста не должен проходить"
    out = compete(insts, queries, sc.index, sc.cfg)
    assert [i.legend_id for i in out] == ["duplex"]


def test_identical_classes_become_ambiguous_with_rivals():
    sc, tpls, qs, (pl,), queries = _setup(
        [("a", sym_slash), ("b", sym_slash)], [(sym_slash, sim(1.0, 10, 400, 300))]
    )
    insts = _instances_at(sc, tpls, qs, pl)
    assert {i.legend_id for i in insts} == {"a", "b"}
    out = compete(insts, queries, sc.index, sc.cfg)
    assert {i.legend_id for i in out} == {"a", "b"}
    assert all(i.status == "ambiguous" for i in out)
    by = {i.legend_id: i for i in out}
    assert "b" in by["a"].rivals and "a" in by["b"].rivals


def test_non_overlapping_instances_untouched():
    sc, tpls, qs, (p1, p2), queries = _setup(
        [("slash", sym_slash), ("x", sym_xcircle)],
        [(sym_slash, sim(1.0, 0, 400, 300)), (sym_xcircle, sim(1.0, 0, 400, 450))],
    )
    insts = _instances_at(sc, tpls, qs, p1)[:1]
    # экземпляр x на втором месте (другая область): конкуренции нет
    tx, qx = tpls[1], qs[1]
    ix = verify_hypothesis(qx, make_hyp("x", true_T(qx, tx, p2)), sc.index, sc.cfg)
    assert ix is not None
    insts.append(ix)
    out = compete(insts, queries, sc.index, sc.cfg)
    assert sorted(i.legend_id for i in out) == ["slash", "x"]
    assert all(i.status == "ok" for i in out)


def test_same_class_overlapping_instances_are_not_touched_by_compete():
    sc, tpls, qs, (pl,), queries = _setup(
        [("slash", sym_slash)], [(sym_slash, sim(1.0, 0, 400, 300))]
    )
    inst = _instances_at(sc, tpls, qs, pl)[0]
    twin = dataclasses.replace(inst, score=inst.score - 0.01)
    out = compete([inst, twin], queries, sc.index, sc.cfg)
    assert len(out) == 2  # дубли своего класса убирает resolve, не compete


def test_empty_and_single_instance_lists():
    sc, tpls, qs, (pl,), queries = _setup(
        [("slash", sym_slash)], [(sym_slash, sim(1.0, 0, 400, 300))]
    )
    assert compete([], queries, sc.index, sc.cfg) == []
    inst = _instances_at(sc, tpls, qs, pl)[0]
    out = compete([inst], queries, sc.index, sc.cfg)
    assert len(out) == 1 and out[0].status == "ok"
    assert isinstance(out[0], Instance)
