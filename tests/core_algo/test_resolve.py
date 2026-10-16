"""3.7 Глобальное разрешение конфликтов за куски."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Instance, Similarity
from symspot.query.extract import extract_query
from symspot.resolve.global_resolve import query_complexity, resolve

from .helpers import Scene, sim, sym_duplex, sym_gfci, sym_slash, sym_xcircle

CFG = Config()


def _queries():
    sc = Scene()
    pls = [
        sc.place("slash", sym_slash(), sim(1.0, 0, 100, 100)),
        sc.place("x", sym_xcircle(), sim(1.0, 0, 250, 100)),
        sc.place("duplex", sym_duplex(), sim(1.0, 0, 400, 100)),
        sc.place("gfci", sym_gfci(), sim(1.0, 0, 100, 250)),
    ]
    sc.build()
    qs = [extract_query(sc.spec(p), sc.index, sc.cfg) for p in pls]
    return {q.legend_id: [q] for q in qs}


QUERIES = _queries()


def _inst(legend, pieces, score=0.95, bbox=(0.0, 0.0, 10.0, 10.0), **kw) -> Instance:
    return Instance(
        legend_id=legend,
        T=Similarity(),
        score=score,
        coverage=score,
        piece_ids=list(pieces),
        parent_ids=sorted({p // 10 for p in pieces}),
        bbox=bbox,
        **kw,
    )


def test_two_instances_of_one_class_in_the_same_place_collapse():
    a = _inst("slash", range(1, 11), score=0.92)
    b = _inst("slash", range(1, 11), score=0.97)
    out = resolve([a, b], QUERIES, CFG)
    assert len(out) == 1
    assert out[0].score == 0.97


def test_near_duplicate_with_high_piece_iou_collapses():
    a = _inst("slash", range(1, 11), score=0.95)
    b = _inst("slash", range(2, 12), score=0.90)  # IoU множеств 9/11
    out = resolve([b, a], QUERIES, CFG)
    assert len(out) == 1 and out[0].score == 0.95


def test_same_class_with_small_overlap_both_kept():
    a = _inst("slash", range(1, 11))
    b = _inst("slash", range(8, 18))  # общих 3 из 10: занято 30 % < 50 %
    out = resolve([a, b], QUERIES, CFG)
    assert len(out) == 2


def test_other_class_with_more_than_half_pieces_taken_is_rejected():
    big = _inst("x", range(1, 21), score=0.9)
    small = _inst("slash", range(1, 9), score=0.99)  # все куски заняты X
    out = resolve([small, big], QUERIES, CFG)
    assert [i.legend_id for i in out] == ["x"]


def test_rejection_does_not_depend_on_input_order():
    big = _inst("x", range(1, 21), score=0.9)
    small = _inst("slash", range(1, 9), score=0.99)
    a = resolve([big, small], QUERIES, CFG)
    b = resolve([small, big], QUERIES, CFG)
    assert [i.legend_id for i in a] == [i.legend_id for i in b] == ["x"]


def test_other_class_with_less_than_half_taken_is_kept():
    big = _inst("x", range(1, 21))
    small = _inst("slash", [18, 19, 20, 21, 22, 23, 24, 25])  # заняты 3 из 8
    out = resolve([big, small], QUERIES, CFG)
    assert sorted(i.legend_id for i in out) == ["slash", "x"]


def test_disjoint_instances_all_kept_and_ids_are_consecutive():
    insts = [
        _inst("slash", range(1, 9)),
        _inst("x", range(101, 121)),
        _inst("duplex", range(201, 211)),
        _inst("slash", range(301, 309)),
    ]
    out = resolve(insts, QUERIES, CFG)
    assert len(out) == 4
    assert sorted(i.id for i in out) == list(range(4))


def test_ids_are_consecutive_after_removals():
    insts = [
        _inst("x", range(1, 21)),
        _inst("slash", range(1, 9)),  # отклонен
        _inst("x", range(101, 121)),
        _inst("x", range(101, 121), score=0.8),  # дубль
    ]
    out = resolve(insts, QUERIES, CFG)
    assert len(out) == 2
    assert sorted(i.id for i in out) == [0, 1]


def test_empty_input():
    assert resolve([], QUERIES, CFG) == []


def test_ambiguous_status_is_preserved():
    a = _inst("slash", range(1, 9), status="ambiguous", rivals=["x"])
    out = resolve([a], QUERIES, CFG)
    assert len(out) == 1 and out[0].status == "ambiguous"


def test_instance_without_pieces_does_not_crash():
    out = resolve([_inst("slash", [])], QUERIES, CFG)
    assert len(out) in (0, 1)


def test_complexity_orders_by_size_and_text_counts():
    c = {k: query_complexity(v[0]) for k, v in QUERIES.items()}
    assert c["x"] > c["slash"]
    assert c["gfci"] > c["duplex"], (
        "текст добавляет сложность (считается линией длиной в диагональ)"
    )
