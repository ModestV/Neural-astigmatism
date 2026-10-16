"""3.5а Отличающие части похожих эталонов."""

from __future__ import annotations

import numpy as np
import pytest

from symspot.core.types import Arc, Query
from symspot.query.discriminative import apply_discriminative_weights, best_alignment
from symspot.query.extract import extract_query

from .helpers import (
    Scene,
    sim,
    sym_duplex,
    sym_gfci,
    sym_slash,
    sym_square,
    sym_tri,
    sym_xcircle,
)


def _queries(*items):
    """items: (legend_id, local_fn). Шаблоны в разных местах одного чертежа."""
    sc = Scene()
    pls = []
    for k, (name, fn) in enumerate(items):
        pls.append(sc.place(name, fn(), sim(1.0, 0, 100 + 150 * k, 100)))
    sc.build()
    return [extract_query(sc.spec(p), sc.index, sc.cfg) for p in pls], sc


def _by_id(qs):
    return {q.legend_id: q for q in qs}


def _diag_masks(q: Query):
    """Сэмплы эталона: круг (сэмплы дуг), диагональ x=y, диагональ x=-y (нормализовано).

    Круг определяется по типу примитива, а не по радиусу: концы диагоналей тоже лежат у
    окружности, но это сэмплы диагоналей с другим направлением касательной.
    """
    is_arc = np.array([isinstance(q.prims[k], Arc) for k in q.sample_prim], dtype=bool)
    ctr = is_arc
    d1 = (~ctr) & (np.abs(q.samples[:, 0] - q.samples[:, 1]) < 1e-6)
    d2 = (~ctr) & (np.abs(q.samples[:, 0] + q.samples[:, 1]) < 1e-6)
    return ctr, d1, d2


def test_second_diagonal_of_xcircle_gets_higher_weight():
    qs, sc = _queries(("slash", sym_slash), ("x", sym_xcircle))
    out = _by_id(apply_discriminative_weights(qs, sc.cfg))
    x = out["x"]
    ctr, d1, d2 = _diag_masks(x)
    assert ctr.sum() > 10 and d1.sum() > 5 and d2.sum() > 5
    assert np.all(x.weights[ctr] == 1.0), "круг есть и у ⊘: он не отличающая часть"
    # вдали от центра и от круга (там линии допустимо считать покрытыми по допуску)
    r = np.hypot(x.samples[:, 0], x.samples[:, 1])
    far = (r > 0.06) & (np.abs(r - 10.0 / x.diag) > 0.06)
    w1, w2 = x.weights[d1 & far], x.weights[d2 & far]
    assert len(w1) > 5 and len(w2) > 5
    # одна диагональ совпадает с диагональю ⊘ (вес 1), другая отличающая (вес > 1)
    lo, hi = sorted([w1, w2], key=lambda w: w.mean())
    assert np.all(lo == 1.0)
    assert np.all(hi > 1.0)


def test_discriminative_weight_equals_config_value():
    qs, sc = _queries(("slash", sym_slash), ("x", sym_xcircle))
    out = _by_id(apply_discriminative_weights(qs, sc.cfg))
    assert out["x"].weights.max() == pytest.approx(sc.cfg.discriminative_weight)
    assert out["x"].weights.min() == pytest.approx(1.0)


def test_subset_query_keeps_unit_weights():
    qs, sc = _queries(("slash", sym_slash), ("x", sym_xcircle))
    out = _by_id(apply_discriminative_weights(qs, sc.cfg))
    np.testing.assert_allclose(out["slash"].weights, 1.0)


def test_unrelated_square_keeps_unit_weights():
    # квадрат со стороной 28 (диагональ 1.36 от ⊘): размер схожий, форма другая
    qs, sc = _queries(("slash", sym_slash), ("x", sym_xcircle), ("sq", sym_square))
    out = _by_id(apply_discriminative_weights(qs, sc.cfg))
    np.testing.assert_allclose(out["sq"].weights, 1.0)
    np.testing.assert_allclose(out["slash"].weights, 1.0)


def test_much_larger_square_is_ignored_by_size_gate():
    qs, sc = _queries(("slash", sym_slash), ("sq", lambda: sym_square(30.0)))
    out = _by_id(apply_discriminative_weights(qs, sc.cfg))
    np.testing.assert_allclose(out["sq"].weights, 1.0)
    np.testing.assert_allclose(out["slash"].weights, 1.0)


def test_chiral_triangle_is_not_similar_to_slashed_circle():
    qs, sc = _queries(("slash", sym_slash), ("tri", sym_tri))
    out = _by_id(apply_discriminative_weights(qs, sc.cfg))
    np.testing.assert_allclose(out["tri"].weights, 1.0)
    np.testing.assert_allclose(out["slash"].weights, 1.0)


def test_single_query_is_unchanged():
    qs, sc = _queries(("slash", sym_slash))
    out = apply_discriminative_weights(qs, sc.cfg)
    assert len(out) == 1
    np.testing.assert_allclose(out[0].weights, 1.0)


def test_empty_list():
    from symspot.core.config import Config

    assert apply_discriminative_weights([], Config()) == []


def test_input_queries_are_not_mutated_and_order_preserved():
    qs, sc = _queries(("slash", sym_slash), ("x", sym_xcircle))
    before = [q.weights.copy() for q in qs]
    out = apply_discriminative_weights(qs, sc.cfg)
    assert [q.legend_id for q in out] == [q.legend_id for q in qs]
    for q, w in zip(qs, before, strict=True):
        np.testing.assert_array_equal(q.weights, w)
    assert all(q.weights.shape == (len(q.samples),) for q in out)
    # геометрия не меняется
    for a, b in zip(qs, out, strict=True):
        np.testing.assert_allclose(a.samples, b.samples)


def test_identical_queries_have_no_discriminative_parts():
    qs, sc = _queries(("a", sym_slash), ("b", sym_slash))
    out = apply_discriminative_weights(qs, sc.cfg)
    for q in out:
        np.testing.assert_allclose(q.weights, 1.0)


def test_duplex_subset_of_gfci_keeps_unit_weights():
    qs, sc = _queries(("duplex", sym_duplex), ("gfci", sym_gfci))
    out = _by_id(apply_discriminative_weights(qs, sc.cfg))
    # геометрия линий GFCI совпадает с duplex, отличается только текст: все линии duplex
    # покрыты GFCI, повышать вес нечему; линии GFCI покрыты duplex
    np.testing.assert_allclose(out["duplex"].weights, 1.0)
    np.testing.assert_allclose(out["gfci"].weights, 1.0)


def test_best_alignment_of_duplex_and_gfci_is_exact():
    qs, sc = _queries(("duplex", sym_duplex), ("gfci", sym_gfci))
    duplex, gfci = qs
    # одинаковые линии: правильное совмещение дает почти нулевой chamfer (поворот не должен
    # теряться из-за вращательной симметрии круга)
    T, sim = best_alignment(gfci, duplex, sc.cfg)
    assert sim > 0.99, f"сходство {sim:.4f}, T={T}"


def test_best_alignment_finds_subset_as_similar():
    qs, sc = _queries(("slash", sym_slash), ("x", sym_xcircle))
    T, sim = best_alignment(qs[1], qs[0], sc.cfg)
    assert sim >= sc.cfg.similar_cover
    # ⊘ как часть круга с крестом: совмещение близко к тождественному по центру
    assert np.hypot(T.tx, T.ty) < 0.05


def test_best_alignment_of_unrelated_shapes_is_not_similar():
    qs, sc = _queries(("slash", sym_slash), ("sq", lambda: sym_square(30.0)))
    _, sim = best_alignment(qs[0], qs[1], sc.cfg)
    assert sim < sc.cfg.similar_cover
