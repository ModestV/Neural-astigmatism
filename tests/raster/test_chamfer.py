from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from symspot.core.config import Config
from symspot.core.types import Hypothesis, Query
from symspot.raster.chamfer import chamfer_search
from symspot.raster.preprocess import RasterPrep, preprocess
from symspot.raster.verify_raster import extract_raster_query
from tests.raster.synth_sheet import SYMBOLS, Copy, center_of, make_sheet

CENTER_TOL = 0.15  # доли диагонали эталона
ANGLE_TOL = math.radians(15.0)

ROTS = [0.0, 37.0, 90.0, 200.0, 300.0, 150.0]
SCALES = [0.85, 1.0, 1.15, 0.9, 1.1, 1.0]
POS = [(150, 150), (400, 160), (250, 330), (500, 330), (150, 400), (570, 120)]
LEGEND = {"slash_circle": (100, 700), "tri_stem": (300, 760)}


def _copies() -> list[Copy]:
    out = []
    for name, dx in (("slash_circle", 0), ("tri_stem", 830)):
        for i in range(6):
            mirror = name == "tri_stem" and i == 4  # одна зеркальная копия
            out.append(Copy(name, SCALES[i], ROTS[i], mirror, (POS[i][0] + dx, POS[i][1])))
    return out


@dataclass
class Scene:
    prep: RasterPrep
    queries: dict[str, Query]
    copies: list[Copy]
    cfg: Config


@pytest.fixture(scope="module")
def scene() -> Scene:
    cfg = Config()
    copies = _copies()
    img, specs = make_sheet(copies, LEGEND)
    prep = preprocess(img, cfg)
    queries = {n: extract_raster_query(s, prep, cfg) for n, s in specs.items()}
    return Scene(prep, queries, copies, cfg)


@pytest.fixture(scope="module")
def hyps(scene: Scene) -> dict[str, list[Hypothesis]]:
    return {n: chamfer_search(q, scene.prep, scene.cfg) for n, q in scene.queries.items()}


def _ang_diff_mod(a: float, b: float, period: float) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def _matches_pose(h: Hypothesis, c: Copy, symmetric: bool) -> bool:
    """Поза гипотезы соответствует копии (для симметричного символа с учетом симметрии:
    круг с косой чертой переходит в себя при повороте на 180 градусов, а отражение равно
    повороту на 90 градусов)."""
    phi = math.radians(c.theta_deg)
    if symmetric:
        shift = 0.0 if h.T.mirror == c.mirror else math.pi / 2
        return _ang_diff_mod(h.T.theta, phi + shift, math.pi) < ANGLE_TOL
    return h.T.mirror == c.mirror and _ang_diff_mod(h.T.theta, phi, 2 * math.pi) < ANGLE_TOL


@pytest.mark.parametrize("name", list(SYMBOLS))
def test_every_copy_found_with_pose(scene: Scene, hyps, name: str) -> None:
    q = scene.queries[name]
    symmetric = SYMBOLS[name]["symmetric"]
    for c in (c for c in scene.copies if c.name == name):
        ctr = center_of(c)
        near = [
            h
            for h in hyps[name]
            if math.hypot(h.T.tx - ctr[0], h.T.ty - ctr[1]) < CENTER_TOL * q.diag
        ]
        assert near, f"{name}: нет гипотезы у копии {c}"
        assert any(_matches_pose(h, c, symmetric) for h in near), f"{name}: неверная поза у {c}"


@pytest.mark.parametrize("name", list(SYMBOLS))
def test_scale_of_best_hypothesis(scene: Scene, hyps, name: str) -> None:
    q = scene.queries[name]
    symmetric = SYMBOLS[name]["symmetric"]
    for c in (c for c in scene.copies if c.name == name):
        ctr = center_of(c)
        good = [
            h
            for h in hyps[name]
            if math.hypot(h.T.tx - ctr[0], h.T.ty - ctr[1]) < CENTER_TOL * q.diag
            and _matches_pose(h, c, symmetric)
        ]
        best = max(good, key=lambda h: h.votes)
        assert best.T.s / q.diag == pytest.approx(c.scale, rel=0.2)


def test_hypothesis_fields_and_limit(scene: Scene, hyps) -> None:
    for name, hs in hyps.items():
        assert 0 < len(hs) <= scene.cfg.raster_topk
        for h in hs:
            assert h.source == "chamfer"
            assert h.legend_id == name
            assert h.votes <= 0.0
            assert -math.pi <= h.T.theta < math.pi
            rel = h.T.s / scene.queries[name].diag
            assert 0.5 * scene.cfg.scale_min <= rel <= 1.5 * scene.cfg.scale_max
        votes = [h.votes for h in hs]
        assert votes == sorted(votes, reverse=True)


def test_topk_limits_count(scene: Scene) -> None:
    cfg = scene.cfg.model_copy(update={"raster_topk": 5})
    hs = chamfer_search(scene.queries["tri_stem"], scene.prep, cfg)
    assert len(hs) == 5


def test_variant_is_propagated(scene: Scene) -> None:
    q = scene.queries["tri_stem"]
    q2 = Query(**{**q.__dict__, "variant": 3})
    hs = chamfer_search(q2, scene.prep, scene.cfg)
    assert hs and all(h.variant == 3 for h in hs)


def test_no_mirror_when_disabled(scene: Scene) -> None:
    cfg = scene.cfg.model_copy(update={"allow_mirror": False})
    hs = chamfer_search(scene.queries["tri_stem"], scene.prep, cfg)
    assert hs and not any(h.T.mirror for h in hs)


def test_empty_sheet_gives_no_hypotheses(scene: Scene) -> None:
    blank = np.full((600, 800), 255, dtype=np.uint8)
    prep = preprocess(blank, scene.cfg)
    assert chamfer_search(scene.queries["tri_stem"], prep, scene.cfg) == []


def test_sheet_without_symbols_has_no_good_hypotheses(scene: Scene) -> None:
    img, _ = make_sheet([], {})
    prep = preprocess(img, scene.cfg)
    hs = chamfer_search(scene.queries["tri_stem"], prep, scene.cfg)
    assert len(hs) <= scene.cfg.raster_topk
    # только линии листа: ни одна гипотеза не должна иметь малый chamfer с шаблоном-треугольником
    assert all(-h.votes > 0.03 for h in hs)


def test_degenerate_query_gives_no_hypotheses(scene: Scene) -> None:
    q = scene.queries["tri_stem"]
    tiny = Query(**{**q.__dict__, "samples": q.samples[:2], "tangents": q.tangents[:2]})
    assert chamfer_search(tiny, scene.prep, scene.cfg) == []


def test_coordinates_follow_image_shift(scene: Scene, hyps) -> None:
    """Сдвиг листа в большем холсте сдвигает гипотезы на то же смещение (координаты исходного
    изображения, а не уменьшенного)."""
    dx, dy = 123, 77
    img, _ = make_sheet(_copies(), LEGEND)
    big = np.full((img.shape[0] + 200, img.shape[1] + 300), 255, dtype=np.uint8)
    big[dy : dy + img.shape[0], dx : dx + img.shape[1]] = img
    prep = preprocess(big, scene.cfg)
    name = "tri_stem"
    q = scene.queries[name]
    hs = chamfer_search(q, prep, scene.cfg)
    for c in (c for c in scene.copies if c.name == name):
        ctr = center_of(c) + np.array([dx, dy])
        assert any(
            math.hypot(h.T.tx - ctr[0], h.T.ty - ctr[1]) < CENTER_TOL * q.diag
            and _matches_pose(h, c, False)
            for h in hs
        )
