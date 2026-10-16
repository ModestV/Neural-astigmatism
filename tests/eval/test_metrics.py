from __future__ import annotations

import json

import pytest

from symspot.core.types import GroundTruth, Instance, Similarity
from symspot.eval.metrics import evaluate


def P(cls, bbox, is_query=False):
    return Instance(
        legend_id=cls,
        T=Similarity.identity(),
        score=1.0,
        coverage=1.0,
        piece_ids=[],
        parent_ids=[],
        bbox=bbox,
        is_query=is_query,
    )


def G(cls, bbox, is_query=False):
    return GroundTruth(legend_id=cls, bbox=bbox, is_query=is_query)


A = (0.0, 0.0, 10.0, 10.0)
B = (100.0, 0.0, 110.0, 10.0)
C = (200.0, 0.0, 210.0, 10.0)


def test_perfect_match():
    gt = [G("a", A), G("b", B)]
    pred = [P("b", B), P("a", A)]  # порядок не важен
    m = evaluate(pred, gt)
    assert (m.overall.tp, m.overall.fp, m.overall.fn) == (2, 0, 0)
    assert m.overall.precision == m.overall.recall == m.overall.f1 == 1.0
    assert m.per_class["a"].f1 == 1.0
    assert m.confusion == {"a": {"a": 1}, "b": {"b": 1}}
    assert sorted((i, j) for i, j, _ in m.matches) == [(0, 1), (1, 0)]


def test_class_mismatch_gives_fp_fn_and_confusion():
    m = evaluate([P("b", A), P("a", B)], [G("a", A), G("a", B)])
    assert (m.overall.tp, m.overall.fp, m.overall.fn) == (1, 1, 1)
    assert (m.per_class["a"].tp, m.per_class["a"].fp, m.per_class["a"].fn) == (1, 0, 1)
    assert (m.per_class["b"].tp, m.per_class["b"].fp, m.per_class["b"].fn) == (0, 1, 0)
    assert m.confusion == {"a": {"a": 1, "b": 1}}
    assert m.overall.precision == 0.5 and m.overall.recall == 0.5


def test_missed_and_false_detection():
    m = evaluate([P("a", A), P("c", C)], [G("a", A), G("b", B)])
    assert (m.overall.tp, m.overall.fp, m.overall.fn) == (1, 1, 1)
    assert m.per_class["c"].fp == 1  # ложное
    assert m.per_class["b"].fn == 1  # пропуск
    assert m.confusion["__none__"] == {"c": 1}
    assert m.confusion["b"] == {"__none__": 1}
    assert m.confusion["a"] == {"a": 1}


def test_iou_below_threshold_is_not_matched():
    shifted = (6.0, 0.0, 16.0, 10.0)  # IoU = 40 / 160 = 0.25
    m = evaluate([P("a", shifted)], [G("a", A)])
    assert (m.overall.tp, m.overall.fp, m.overall.fn) == (0, 1, 1)
    assert m.matches == []
    # порог ниже IoU: сопоставляется
    m = evaluate([P("a", shifted)], [G("a", A)], iou_thr=0.2)
    assert m.overall.tp == 1 and m.matches[0][2] == pytest.approx(0.25)


def test_iou_near_threshold_matches():
    half = (5.0, 0.0, 15.0, 10.0)  # IoU = 50 / 150 = 1/3
    assert evaluate([P("a", half)], [G("a", A)], iou_thr=1 / 3 - 1e-9).overall.tp == 1
    assert evaluate([P("a", half)], [G("a", A)], iou_thr=1 / 3 + 1e-9).overall.tp == 0


def test_ignore_queries():
    pred = [P("a", A, is_query=True), P("b", B)]
    gt = [G("a", A, is_query=True), G("b", B)]
    m = evaluate(pred, gt)
    assert (m.overall.tp, m.overall.fp, m.overall.fn) == (1, 0, 0)
    assert "a" not in m.per_class
    assert m.matches[0][:2] == (1, 1)  # индексы в исходных списках
    m = evaluate(pred, gt, ignore_queries=False)
    assert m.overall.tp == 2
    # эталон есть только в GT: при ignore_queries это не пропуск
    m = evaluate([P("b", B)], gt)
    assert (m.overall.fp, m.overall.fn) == (0, 0)
    m = evaluate([P("b", B)], gt, ignore_queries=False)
    assert m.overall.fn == 1


def test_similar_group_error():
    groups = [{"a", "b"}]
    gt = [G("a", A), G("a", B), G("a", C), G("z", (300.0, 0.0, 310.0, 10.0))]
    pred = [
        P("b", A),  # путаница внутри группы
        P("a", B),  # верно
        P("z", C),  # путаница вне группы
        P("a", (300.0, 0.0, 310.0, 10.0)),  # GT z не из группы: в знаменатель не входит
    ]
    m = evaluate(pred, gt, similar_groups=groups)
    assert m.n_similar == 3
    assert m.similar_group_error == pytest.approx(1 / 3)
    # без групп
    m = evaluate(pred, gt)
    assert m.n_similar == 0 and m.similar_group_error == 0.0


def test_similar_group_unmatched_not_counted():
    m = evaluate([], [G("a", A)], similar_groups=[{"a", "b"}])
    assert m.n_similar == 0 and m.similar_group_error == 0.0


def test_empty_inputs():
    m = evaluate([], [])
    assert (m.overall.tp, m.overall.fp, m.overall.fn) == (0, 0, 0)
    assert m.overall.f1 == 0.0 and m.per_class == {} and m.confusion == {}
    assert m.to_dict()["n_similar"] == 0
    m = evaluate([P("a", A)], [])
    assert (m.overall.fp, m.overall.fn) == (1, 0)
    m = evaluate([], [G("a", A)])
    assert (m.overall.fp, m.overall.fn) == (0, 1)


def test_hungarian_beats_greedy():
    g1 = G("a", (0.0, 0.0, 10.0, 10.0))
    g2 = G("a", (4.0, 0.0, 14.0, 10.0))
    pa = P("a", (1.0, 0.0, 11.0, 10.0))  # IoU с g1 = 0.818, с g2 = 0.538
    pb = P("a", (-3.0, 0.0, 7.0, 10.0))  # IoU с g1 = 0.538, с g2 = 0.176
    # жадное по IoU: pa-g1, pb остается без пары (с g2 только 0.176) -> 1 совпадение
    m = evaluate([pa, pb], [g1, g2])
    assert m.overall.tp == 2 and m.overall.fp == 0 and m.overall.fn == 0
    assert {(i, j) for i, j, _ in m.matches} == {(0, 1), (1, 0)}


def test_to_dict_is_json_serializable():
    m = evaluate([P("b", A), P("c", C)], [G("a", A), G("b", B)], similar_groups=[{"a", "b"}])
    d = json.loads(json.dumps(m.to_dict()))
    assert set(d) == {"overall", "per_class", "confusion", "similar_group_error", "n_similar"}
    assert set(d["overall"]) == {"precision", "recall", "f1", "tp", "fp", "fn"}
    assert d["overall"]["fp"] == 2 and d["overall"]["fn"] == 2
    assert d["confusion"]["a"] == {"b": 1}
    assert d["n_similar"] == 1 and d["similar_group_error"] == 1.0
