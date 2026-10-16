"""3.2-3.3 Таблица пар и голосование за преобразование."""

from __future__ import annotations

import math

import numpy as np
import pytest

from symspot.core.config import Config
from symspot.normalize.index import PrimIndex
from symspot.query.extract import extract_query
from symspot.search.pairs import PairTable
from symspot.search.voting import find_candidates
from symspot.verify.coverage import verify_hypothesis

from .helpers import (
    Scene,
    hyp_matches,
    make_hyp,
    sim,
    sym_slash,
    sym_tri,
    true_T,
)

SYMBOLS = {"slash": (sym_slash, True), "tri": (sym_tri, False)}

TRANSFORMS = {
    "rot0": sim(1.0, 0),
    "rot90": sim(1.0, 90),
    "rot37": sim(1.0, 37),
    "rot-125": sim(1.0, -125),
    "scale0.8": sim(0.8, 0),
    "scale1.2": sim(1.2, 0),
    "mirror": sim(1.0, 0, mirror=True),
    "mirror_rot123_scale1.2": sim(1.2, 123, mirror=True),
    "rot-40_scale0.8": sim(0.8, -40),
}


def _candidates(sc: Scene, template, cfg=None):
    cfg = cfg or sc.cfg
    q = extract_query(sc.spec(template), sc.index, cfg)
    table = PairTable.build(sc.index, cfg.pair_radius_factor * q.diag * cfg.scale_max, cfg)
    return q, find_candidates(q, table, sc.index, cfg)


def _scene_with_copy(sym_name: str, S):
    local = SYMBOLS[sym_name][0]()
    sc = Scene()
    t = sc.place(sym_name, local, sim(1.0, 0, 100, 100))
    # стена мимо обеих копий: реалистичный фон
    sc.add_wall((0, 520), (600, 520))
    sc.add_wall((250, 0), (250, 600))
    c = sc.place(sym_name, local, type(S)(S.s, S.theta, 400.0, 300.0, S.mirror))
    sc.build()
    return sc, t, c


@pytest.mark.parametrize("tname", list(TRANSFORMS))
@pytest.mark.parametrize("sym_name", list(SYMBOLS))
def test_finds_transformed_copy(sym_name, tname):
    sc, t, c = _scene_with_copy(sym_name, TRANSFORMS[tname])
    q, hyps = _candidates(sc, t)
    T = true_T(q, t, c)
    symmetric = SYMBOLS[sym_name][1]
    ok = [h for h in hyps if hyp_matches(h, T, q, symmetric=symmetric)]
    assert ok, (
        f"{sym_name}/{tname}: нет гипотезы рядом с истиной {T}; "
        f"получено {len(hyps)}: {[h.T for h in hyps[:5]]}"
    )
    if not symmetric and TRANSFORMS[tname].mirror:
        assert all(h.T.mirror for h in ok)


@pytest.mark.parametrize("sym_name", list(SYMBOLS))
def test_template_itself_is_found(sym_name):
    sc, t, _ = _scene_with_copy(sym_name, sim(1.0, 0))
    q, hyps = _candidates(sc, t)
    T = q.frame
    assert any(hyp_matches(h, T, q, symmetric=SYMBOLS[sym_name][1]) for h in hyps)


@pytest.mark.parametrize("sym_name", list(SYMBOLS))
def test_scale_2_is_out_of_range_and_not_found(sym_name):
    sc, t, c = _scene_with_copy(sym_name, sim(2.0, 20))
    q, hyps = _candidates(sc, t)
    T = true_T(q, t, c)
    near = [h for h in hyps if math.hypot(h.T.tx - T.tx, h.T.ty - T.ty) < 0.5 * T.s]
    accepted = [h for h in near if verify_hypothesis(q, h, sc.index, sc.cfg) is not None]
    assert not accepted, "копия в масштабе 2.0 вне [scale_min, scale_max] принята"
    # никаких гипотез с масштабом правдоподобного размера 2.0
    assert not any(h.T.s / q.diag > sc.cfg.scale_max * 1.1 for h in hyps)


def test_all_hypotheses_respect_scale_range_and_are_sorted():
    sc = Scene()
    t = sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    sc.place("tri", sym_tri(), sim(0.5, 10, 300, 100))
    sc.place("tri", sym_tri(), sim(1.0, 70, 300, 300))
    sc.place("tri", sym_tri(), sim(1.3, -70, 100, 300))
    sc.place("tri", sym_tri(), sim(2.0, 0, 450, 450))
    sc.build()
    q, hyps = _candidates(sc, t)
    assert len(hyps) >= 3
    cfg = sc.cfg
    for h in hyps:
        r = h.T.s / q.diag
        assert cfg.scale_min / 1.1 <= r <= cfg.scale_max * 1.1
        assert h.legend_id == "tri"
    votes = [h.votes for h in hyps]
    assert votes == sorted(votes, reverse=True)


def test_multiple_copies_all_found():
    sc = Scene()
    t = sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    copies = [
        sc.place("tri", sym_tri(), sim(1.0, 90, 300, 100)),
        sc.place("tri", sym_tri(), sim(1.15, 200, 100, 300)),
        sc.place("tri", sym_tri(), sim(0.85, 15, 300, 300, True)),
    ]
    sc.build()
    q, hyps = _candidates(sc, t)
    for c in copies:
        T = true_T(q, t, c)
        assert any(hyp_matches(h, T, q) for h in hyps), f"копия {c.S} не найдена"


def test_max_hypotheses_cap():
    sc = Scene()
    t = sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    for k in range(5):
        sc.place("tri", sym_tri(), sim(1.0, 30 * k, 100 + 100 * k, 300))
    sc.build()
    cfg = Config(max_hypotheses_per_query=2)
    q, hyps = _candidates(sc, t, cfg)
    assert 1 <= len(hyps) <= 2


def test_mirror_disabled_gives_no_mirrored_hypotheses():
    sc = Scene()
    t = sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    c = sc.place("tri", sym_tri(), sim(1.0, 0, 400, 300, True))
    sc.build()
    cfg = Config(allow_mirror=False)
    q, hyps = _candidates(sc, t, cfg)
    assert not any(h.T.mirror for h in hyps)
    T = true_T(q, t, c)
    assert not any(hyp_matches(h, T, q) for h in hyps)


@pytest.mark.parametrize("sym_name", list(SYMBOLS))
def test_hatching_does_not_produce_accepted_instances(sym_name):
    local = SYMBOLS[sym_name][0]()
    sc = Scene()
    t = sc.place(sym_name, local, sim(1.0, 0, 100, 100))
    sc.add_hatch(300, 300, 90, 90)
    sc.build()
    q, hyps = _candidates(sc, t)
    in_hatch = [h for h in hyps if 280 < h.T.tx < 410 and 280 < h.T.ty < 410]
    for h in in_hatch:
        assert verify_hypothesis(q, h, sc.index, sc.cfg) is None, f"штриховка принята: {h.T}"
    # единственное принятое место (если вообще найдено) это сам эталон
    accepted = [h for h in hyps if verify_hypothesis(q, h, sc.index, sc.cfg) is not None]
    assert all(math.hypot(h.T.tx - 100, h.T.ty - 100) < 0.5 * q.diag for h in accepted)


def test_symbol_next_to_hatching_is_still_found():
    sc = Scene()
    t = sc.place("slash", sym_slash(), sim(1.0, 0, 100, 100))
    sc.add_hatch(300, 300, 90, 90)
    c = sc.place("slash", sym_slash(), sim(1.0, 25, 425, 345))  # сразу у края штриховки
    sc.build()
    q, hyps = _candidates(sc, t)
    T = true_T(q, t, c)
    assert any(hyp_matches(h, T, q, symmetric=True) for h in hyps)


def test_empty_drawing_gives_no_hypotheses():
    sc = Scene()
    t = sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    sc.build()
    q = extract_query(sc.spec(t), sc.index, sc.cfg)
    empty = PrimIndex([], sc.cfg.sample_step * q.diag)
    table = PairTable.build(empty, 100.0, sc.cfg)
    assert find_candidates(q, table, empty, sc.cfg) == []


def test_drawing_with_unrelated_lines_gives_no_accepted_hypotheses():
    sc = Scene()
    t = sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    rng = np.random.default_rng(3)
    for _ in range(15):
        p = rng.uniform(250, 550, 2)
        a = rng.uniform(0, 2 * math.pi)
        ln = rng.uniform(20, 80)
        sc.add_segment(p, p + ln * np.array([math.cos(a), math.sin(a)]))
    sc.build()
    q, hyps = _candidates(sc, t)
    far = [h for h in hyps if h.T.tx > 200 or h.T.ty > 200]
    assert all(verify_hypothesis(q, h, sc.index, sc.cfg) is None for h in far)


def test_pair_table_lookup_shapes():
    sc = Scene()
    sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    sc.place("slash", sym_slash(), sim(1.0, 0, 200, 100))
    sc.build()
    table = PairTable.build(sc.index, 100.0, sc.cfg)
    k, row = table.lookup(np.array([], dtype=np.int64))
    assert len(k) == 0 and len(row) == 0
    assert len(table.strokes) >= 6
    assert len(table.a) == len(table.b)
    assert np.all(table.a < table.b)


def test_pair_table_radius_limits_pairs():
    sc = Scene()
    sc.place("tri", sym_tri(), sim(1.0, 0, 100, 100))
    sc.place("tri", sym_tri(), sim(1.0, 0, 500, 500))
    sc.build()
    small = PairTable.build(sc.index, 40.0, sc.cfg)
    big = PairTable.build(sc.index, 2000.0, sc.cfg)
    assert len(small.a) < len(big.a)
    assert len(small.a) > 0


def test_hypothesis_objects_are_well_formed():
    sc, t, c = _scene_with_copy("tri", sim(1.1, 15))
    q, hyps = _candidates(sc, t)
    assert hyps
    for h in hyps:
        assert h.legend_id == "tri"
        assert h.source in ("vote", "exact", "expand", "chamfer")
        assert h.votes > 0
        assert h.T.s > 0
    assert make_hyp("tri", hyps[0].T).T == hyps[0].T
