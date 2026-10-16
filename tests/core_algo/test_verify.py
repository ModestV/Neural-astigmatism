"""3.4 ICP и проверка гипотезы покрытием."""

from __future__ import annotations

import math

import numpy as np
import pytest

from symspot.core.geometry_utils import bbox_iou
from symspot.core.types import QuerySpec, Similarity
from symspot.normalize.index import PrimIndex
from symspot.query.extract import extract_query
from symspot.verify.coverage import verify_hypothesis
from symspot.verify.icp import icp_refine

from .helpers import (
    Scene,
    angle_diff_deg,
    hyp_matches,
    instance_diag,
    make_hyp,
    sim,
    sym_circle,
    sym_slash,
    sym_text_circle,
    sym_tri,
    sym_xcircle,
    true_T,
)


def _perturb(T: Similarity, shift_diag=0.1, deg=5.0, scale=1.05, direction_deg=40.0) -> Similarity:
    d = shift_diag * instance_diag(T)
    a = math.radians(direction_deg)
    return Similarity(
        T.s * scale,
        T.theta + math.radians(deg),
        T.tx + d * math.cos(a),
        T.ty + d * math.sin(a),
        T.mirror,
    )


def _pair(local_fn, S_copy, extra=None):
    sc = Scene()
    t = sc.place("A", local_fn(), sim(1.0, 0, 100, 100))
    c = sc.place("A", local_fn(), S_copy)
    if extra:
        extra(sc)
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    return sc, q, t, c


@pytest.mark.parametrize(
    "S_copy",
    [
        sim(1.0, 0, 400, 300),
        sim(1.1, 33, 400, 300),
        sim(0.85, -100, 400, 300, True),
    ],
    ids=["plain", "rot33_scale1.1", "mirror_rot-100_scale0.85"],
)
@pytest.mark.parametrize("sym", ["slash", "tri"])
def test_verify_recovers_true_transform_from_perturbed_hypothesis(sym, S_copy):
    fn = {"slash": sym_slash, "tri": sym_tri}[sym]
    sc, q, t, c = _pair(fn, S_copy)
    T = true_T(q, t, c)
    for direction in (40.0, 200.0):
        h = make_hyp("A", _perturb(T, direction_deg=direction))
        inst = verify_hypothesis(q, h, sc.index, sc.cfg)
        assert inst is not None, f"верная гипотеза с ошибкой отвергнута (направление {direction})"
        diag = instance_diag(T)
        assert math.hypot(inst.T.tx - T.tx, inst.T.ty - T.ty) < 0.02 * diag
        assert abs(math.log(inst.T.s / T.s)) < 0.03
        assert inst.T.mirror == T.mirror or sym == "slash"
        if sym == "tri":
            assert angle_diff_deg(inst.T.theta, T.theta) < 2.0
        assert hyp_matches(make_hyp("A", inst.T), T, q, symmetric=(sym == "slash"))
        assert inst.legend_id == "A"
        assert inst.score >= sc.cfg.accept_score
        assert 0.0 < inst.coverage <= 1.0 + 1e-9
        assert inst.score <= 1.0 + 1e-9


def test_verified_instance_bbox_is_near_copy_box():
    sc, q, t, c = _pair(sym_tri, sim(1.1, 33, 400, 300))
    T = true_T(q, t, c)
    inst = verify_hypothesis(q, make_hyp("A", _perturb(T)), sc.index, sc.cfg)
    assert inst is not None
    assert bbox_iou(inst.bbox, c.box) > 0.6
    assert inst.bbox[0] <= inst.bbox[2] and inst.bbox[1] <= inst.bbox[3]


def test_exact_hypothesis_scores_near_one():
    sc, q, t, c = _pair(sym_slash, sim(1.0, 0, 400, 300))
    inst = verify_hypothesis(q, make_hyp("A", true_T(q, t, c)), sc.index, sc.cfg)
    assert inst is not None
    assert inst.score > 0.95
    assert inst.coverage > 0.95


def test_circle_without_diagonal_is_rejected_for_slashed_query():
    sc = Scene()
    t = sc.place("slash", sym_slash(), sim(1.0, 0, 100, 100))
    c = sc.place("plain", sym_circle(), sim(1.0, 0, 400, 300))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    T = true_T(q, t, c)
    assert verify_hypothesis(q, make_hyp("slash", T), sc.index, sc.cfg) is None
    # даже с небольшой ошибкой гипотезы
    assert verify_hypothesis(q, make_hyp("slash", _perturb(T)), sc.index, sc.cfg) is None


def test_xcircle_query_rejects_slashed_circle_with_one_diagonal_missing():
    sc = Scene()
    t = sc.place("x", sym_xcircle(), sim(1.0, 0, 100, 100))
    c = sc.place("slash", sym_slash(), sim(1.0, 0, 400, 300))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    assert verify_hypothesis(q, make_hyp("x", true_T(q, t, c)), sc.index, sc.cfg) is None


def test_text_mismatch_rejected_and_text_match_accepted():
    sc = Scene()
    t = sc.place("co", sym_text_circle("CO"), sim(1.0, 0, 100, 100))
    sd = sc.place("sd", sym_text_circle("SD"), sim(1.0, 0, 400, 300))
    co2 = sc.place("co", sym_text_circle("CO"), sim(1.0, 0, 400, 100))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    assert q.texts == ["CO"]
    assert verify_hypothesis(q, make_hyp("co", true_T(q, t, sd)), sc.index, sc.cfg) is None
    inst = verify_hypothesis(q, make_hyp("co", true_T(q, t, co2)), sc.index, sc.cfg)
    assert inst is not None and inst.score >= sc.cfg.accept_score


def test_text_comparison_ignores_case_and_spaces():
    sc = Scene()
    t = sc.place("co", sym_text_circle("CO"), sim(1.0, 0, 100, 100))
    low = sc.place("co", sym_text_circle("c o"), sim(1.0, 0, 400, 300))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    assert verify_hypothesis(q, make_hyp("co", true_T(q, t, low)), sc.index, sc.cfg) is not None


def test_missing_text_rejected():
    sc = Scene()
    t = sc.place("co", sym_text_circle("CO"), sim(1.0, 0, 100, 100))
    bare = sc.place("bare", sym_circle(), sim(1.0, 0, 400, 300))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    assert verify_hypothesis(q, make_hyp("co", true_T(q, t, bare)), sc.index, sc.cfg) is None


def test_extra_line_through_symbol_hardly_lowers_score():
    S = sim(1.0, 20, 400, 300)
    sc0, q0, t0, c0 = _pair(sym_slash, S)
    base = verify_hypothesis(q0, make_hyp("A", true_T(q0, t0, c0)), sc0.index, sc0.cfg)
    assert base is not None

    def add_line(sc):
        sc.add_segment((330, 306), (470, 292))  # линия через символ, не совпадающая с его линиями

    sc, q, t, c = _pair(sym_slash, S, add_line)
    inst = verify_hypothesis(q, make_hyp("A", true_T(q, t, c)), sc.index, sc.cfg)
    assert inst is not None
    assert inst.score >= sc.cfg.accept_score
    assert base.score - inst.score < 0.1


def test_wall_through_symbol_still_accepted():
    def add_wall(sc):
        sc.add_wall((300, 303), (500, 303))

    sc, q, t, c = _pair(sym_tri, sim(1.0, 0, 400, 300), add_wall)
    inst = verify_hypothesis(q, make_hyp("A", true_T(q, t, c)), sc.index, sc.cfg)
    assert inst is not None and inst.score >= sc.cfg.accept_score


def test_hypothesis_on_empty_place_is_rejected():
    sc, q, t, c = _pair(sym_slash, sim(1.0, 0, 400, 300))
    T = Similarity(q.frame.s, 0.0, 520.0, 520.0, False)
    assert verify_hypothesis(q, make_hyp("A", T), sc.index, sc.cfg) is None


def test_verify_on_empty_index_returns_none():
    sc, q, t, c = _pair(sym_slash, sim(1.0, 0, 400, 300))
    empty = PrimIndex([], sc.cfg.sample_step * q.diag)
    assert verify_hypothesis(q, make_hyp("A", q.frame), empty, sc.cfg) is None


def test_wrong_scale_hypothesis_is_rejected_or_corrected():
    sc, q, t, c = _pair(sym_tri, sim(1.0, 0, 400, 300))
    T = true_T(q, t, c)
    bad = Similarity(T.s * 1.5, T.theta, T.tx, T.ty, T.mirror)
    inst = verify_hypothesis(q, make_hyp("A", bad), sc.index, sc.cfg)
    # допустимо и отвергнуть, и (маловероятно) исправить масштаб; но не принимать с неверным T
    if inst is not None:
        assert abs(math.log(inst.T.s / T.s)) < 0.05


# --- ICP напрямую ---


def test_icp_converges_to_true_transform_and_small_residual():
    sc, q, t, c = _pair(sym_tri, sim(1.1, 33, 400, 300))
    T = true_T(q, t, c)
    T0 = _perturb(T)
    T1, res = icp_refine(q, T0, sc.index, sc.cfg)
    d0 = math.hypot(T0.tx - T.tx, T0.ty - T.ty)
    d1 = math.hypot(T1.tx - T.tx, T1.ty - T.ty)
    assert d1 < d0
    assert d1 < 0.02 * instance_diag(T)
    assert res < 0.01
    assert T1.mirror == T.mirror


def test_icp_keeps_mirror_flag():
    sc, q, t, c = _pair(sym_tri, sim(1.0, 0, 400, 300, True))
    T = true_T(q, t, c)
    assert T.mirror
    T1, res = icp_refine(q, _perturb(T), sc.index, sc.cfg)
    assert T1.mirror
    assert res < 0.02


def test_icp_exact_start_stays_put():
    sc, q, t, c = _pair(sym_slash, sim(1.0, 0, 400, 300))
    T = true_T(q, t, c)
    T1, res = icp_refine(q, T, sc.index, sc.cfg)
    assert math.hypot(T1.tx - T.tx, T1.ty - T.ty) < 0.005 * instance_diag(T)
    assert res < 0.005


def test_icp_without_matches_gives_infinite_residual():
    sc, q, t, c = _pair(sym_slash, sim(1.0, 0, 400, 300))
    far = Similarity(q.frame.s, 0.0, 550.0, 550.0, False)
    T1, res = icp_refine(q, far, sc.index, sc.cfg)
    assert math.isinf(res)
    np.testing.assert_allclose([T1.tx, T1.ty], [far.tx, far.ty], atol=1e-6)


def test_icp_on_empty_index_gives_infinite_residual():
    sc, q, t, c = _pair(sym_slash, sim(1.0, 0, 400, 300))
    empty = PrimIndex([], sc.cfg.sample_step * q.diag)
    T1, res = icp_refine(q, q.frame, empty, sc.cfg)
    assert math.isinf(res)


def test_icp_matches_parallel_but_not_perpendicular_lines():
    # эталон: один вертикальный отрезок; чертеж: горизонтальный отрезок на том же месте
    q_sc = Scene()
    q_sc.add_segment((100, 90), (100, 110))
    q_sc.build()
    q = extract_query(QuerySpec("bar", (95, 85, 105, 115)), q_sc.index, q_sc.cfg)
    h_sc = Scene()
    h_sc.add_segment((90, 100), (110, 100))
    h_sc.build()
    _, res = icp_refine(q, q.frame, h_sc.index, h_sc.cfg)
    assert math.isinf(res), "перпендикулярные линии не должны сопоставляться"
    v_sc = Scene()
    v_sc.add_segment((101, 90), (101, 110))
    v_sc.build()
    T1, res = icp_refine(q, q.frame, v_sc.index, v_sc.cfg)
    assert math.isfinite(res)
