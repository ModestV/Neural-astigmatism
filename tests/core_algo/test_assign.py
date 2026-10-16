"""3.6 Назначение кусков экземпляру."""

from __future__ import annotations

import pytest

from symspot.core.geometry_utils import bbox_diag, bbox_expand, prim_bbox
from symspot.paint.assign import assign_pieces
from symspot.query.extract import extract_query
from symspot.verify.coverage import verify_hypothesis

from .helpers import (
    Scene,
    make_hyp,
    sim,
    sym_slash,
    sym_text_circle,
    sym_tri,
    true_T,
)

SYMS = {"slash": sym_slash, "tri": sym_tri, "co": lambda: sym_text_circle("CO")}


def _setup(sym, S, walls=(), extra=None):
    sc = Scene()
    t = sc.place("A", SYMS[sym](), sim(1.0, 0, 100, 100))
    c = sc.place("A", SYMS[sym](), S)
    for w in walls:
        sc.add_wall(*w)
    if extra:
        extra(sc)
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    inst = verify_hypothesis(q, make_hyp("A", true_T(q, t, c)), sc.index, sc.cfg)
    assert inst is not None, "предусловие: экземпляр должен пройти проверку"
    return sc, q, c, inst


def _contained(sc, ids, box, pad_frac):
    big = bbox_expand(box, pad_frac * bbox_diag(box))
    by_id = sc.drawing.prim_by_id()
    for i in ids:
        b = prim_bbox(by_id[i])
        if not (big[0] <= b[0] and big[1] <= b[1] and b[2] <= big[2] and b[3] <= big[3]):
            return False, i
    return True, None


@pytest.mark.parametrize("sym", ["slash", "tri"])
@pytest.mark.parametrize("dy", [0.0, 4.0, -6.0])
def test_wall_through_symbol_only_symbol_pieces_assigned(sym, dy):
    S = sim(1.0, 25, 400, 300)
    sc, q, c, inst = _setup(sym, S, walls=[((250, 300 + dy), (560, 300 + dy))])
    out = assign_pieces(inst, q, sc.index, sc.cfg)
    wall = set(sc.wall_pieces())
    sym_pieces = set(sc.pieces_of(c.parents))
    assert wall and sym_pieces
    got = set(out.piece_ids)
    assert not (got & wall), "куски стены назначены символу"
    assert sym_pieces <= got, f"не все линии символа назначены, пропущено {sym_pieces - got}"
    ok, bad = _contained(sc, out.piece_ids, inst.bbox, 0.05)
    assert ok, f"кусок {bad} вышел за рамку экземпляра"
    assert set(out.parent_ids) == set(c.parents)
    assert not (set(out.parent_ids) & set(sc.wall_parents))


def test_wall_pieces_inside_circle_are_not_assigned():
    sc, q, c, inst = _setup("slash", sim(1.0, 0, 400, 300), walls=[((250, 304), (560, 304))])
    out = assign_pieces(inst, q, sc.index, sc.cfg)
    by_id = sc.drawing.prim_by_id()
    inside = [
        i
        for i in sc.wall_pieces()
        if abs(by_id[i].p0[0] - 400) < 12 and abs(by_id[i].p1[0] - 400) < 12  # type: ignore[union-attr]
    ]
    assert inside, "ожидался хотя бы один кусок стены внутри круга"
    assert not (set(inside) & set(out.piece_ids))


def test_clean_symbol_gets_exactly_its_pieces():
    sc, q, c, inst = _setup("tri", sim(0.9, -60, 400, 300, True))
    out = assign_pieces(inst, q, sc.index, sc.cfg)
    assert set(out.piece_ids) == set(sc.pieces_of(c.parents))
    assert len(out.piece_ids) == len(set(out.piece_ids))


def test_neighbouring_parallel_line_is_not_assigned():
    def extra(sc):
        # прямая параллельно стороне, но заметно дальше допуска покрытия
        sc.add_segment((450, 250), (450, 350))

    sc, q, c, inst = _setup("slash", sim(1.0, 0, 400, 300), extra=extra)
    out = assign_pieces(inst, q, sc.index, sc.cfg)
    foreign = {p.id for p in sc.drawing.primitives if p.parent_id == sc.prims[-1].id}
    assert foreign and not (foreign & set(out.piece_ids))


def test_text_of_instance_is_assigned_and_foreign_text_is_not():
    def extra(sc):
        sc.add_text("XX", (460, 295, 480, 305))

    sc, q, c, inst = _setup("co", sim(1.0, 0, 400, 300), extra=extra)
    out = assign_pieces(inst, q, sc.index, sc.cfg)
    by_id = sc.drawing.prim_by_id()
    texts = [i for i in out.piece_ids if hasattr(by_id[i], "text")]
    assert [by_id[i].text for i in texts] == ["CO"]


def test_assign_keeps_score_and_class():
    sc, q, c, inst = _setup("slash", sim(1.0, 0, 400, 300))
    score, legend, T = inst.score, inst.legend_id, inst.T
    out = assign_pieces(inst, q, sc.index, sc.cfg)
    assert out.score == pytest.approx(score)
    assert out.legend_id == legend
    assert out.T == T
