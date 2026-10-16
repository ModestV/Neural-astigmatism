from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from symspot.core.geometry_utils import prim_bbox, prims_bbox, transform_prim
from symspot.core.types import TextPrim
from symspot.synth.__main__ import main as cli_main
from symspot.synth.elements import box_intersects
from symspot.synth.export import export_plan, generate, plan_to_svg
from symspot.synth.plan_generator import SynthPlan, generate_plan
from symspot.synth.symbols import similar_groups, symbol_library

SVG_NS = "{http://www.w3.org/2000/svg}"
TOL = 1e-6


@lru_cache(maxsize=8)
def _plan(seed: int, size: str = "small") -> SynthPlan:
    return generate_plan(seed, size)  # type: ignore[arg-type]


def test_library_has_groups() -> None:
    lib = symbol_library()
    assert len(lib) >= 12
    groups = {
        "receptacle": {"duplex", "gfci", "r220"},
        "switch": {"switch_s", "switch_s3", "switch_sd"},
        "light": {"light_recessed", "light_surface"},
        "detector": {"det_smoke", "det_co"},
    }
    for g, ids in groups.items():
        assert {k for k, s in lib.items() if s.group == g} == ids
    assert {"fan", "junction"} <= {k for k, s in lib.items() if s.group is None}
    assert len(similar_groups(lib)) == 4
    for s in lib.values():
        b = prims_bbox(s.prims)
        assert b is not None
        assert abs((b[0] + b[2]) / 2) < 1e-9 and abs((b[1] + b[3]) / 2) < 1e-9
        assert 0.7 < s.diag < 1.6


def test_determinism(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    export_plan(generate_plan(5, "small"), a, "x", raster=False)
    export_plan(generate_plan(5, "small"), b, "x", raster=False)
    export_plan(generate_plan(6, "small"), c, "x", raster=False)
    for f in ("x.svg", "x_gt.json", "x_coco.json"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    assert (a / "x.svg").read_bytes() != (c / "x.svg").read_bytes()
    assert (a / "x_gt.json").read_bytes() != (c / "x_gt.json").read_bytes()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_counts(seed: int) -> None:
    plan = _plan(seed)
    assert 1400 <= plan.n_prims <= 2600
    assert 40 <= len(plan.plan_instances) <= 80
    assert {i.legend_id for i in plan.instances if i.is_query} == set(symbol_library())
    assert len(plan.queries) == len(symbol_library())


def test_gt_bbox_contains_geometry() -> None:
    lib = symbol_library()
    for seed in (0, 1):
        plan = _plan(seed)
        for inst in plan.instances:
            tprims = [transform_prim(p, inst.T) for p in lib[inst.legend_id].prims]
            tight = prims_bbox(tprims)
            assert tight is not None
            bb = inst.bbox
            assert bb[0] < tight[0] and bb[1] < tight[1] and bb[2] > tight[2] and bb[3] > tight[3]
            pad = (tight[0] - bb[0], tight[1] - bb[1])
            diag = math.hypot(tight[2] - tight[0], tight[3] - tight[1])
            assert pad[0] == pytest.approx(0.08 * diag, abs=1e-6)
            for e in inst.elements:
                b = prim_bbox(e.prim)
                assert b[0] >= bb[0] - TOL and b[1] >= bb[1] - TOL
                assert b[2] <= bb[2] + TOL and b[3] <= bb[3] + TOL


def test_mirror_only_without_text() -> None:
    lib = symbol_library()
    for inst in _plan(0).instances:
        if lib[inst.legend_id].has_text:
            assert not inst.T.mirror
        assert 0.85 * 0.99 < inst.T.s / (15.0 / lib[inst.legend_id].diag) < 1.15 * 1.01


def test_all_hard_tags_present() -> None:
    tags: set[str] = set()
    for seed in (0, 1, 2):
        for inst in _plan(seed).plan_instances:
            tags.update(inst.hard)
    assert tags == {"wall_through", "near_hatch", "split_lines", "bezier_circle"}


def test_hard_tags_match_geometry() -> None:
    plan = _plan(0)
    for inst in plan.plan_instances:
        if "bezier_circle" in inst.hard:
            assert any(e.bezier for e in inst.elements)
        if "split_lines" in inst.hard:
            lib_segs = sum(
                1 for p in plan.library[inst.legend_id].prims if p.__class__.__name__ == "Segment"
            )
            assert len(inst.elements) > len(plan.library[inst.legend_id].prims) or lib_segs == 0


def test_svg_structure() -> None:
    plan = _plan(0)
    root = ET.fromstring(plan_to_svg(plan))
    assert root.get("width") == "600pt" and root.get("height") == "400pt"
    assert root.get("viewBox") == "0 0 600 400"
    items = list(root)
    assert len(items) == plan.n_prims
    ids = [el.get("id") for el in items]
    assert len(set(ids)) == len(ids)
    paths = [el for el in items if el.tag == SVG_NS + "path"]
    texts = [el for el in items if el.tag == SVG_NS + "text"]
    assert paths and texts and len(paths) + len(texts) == len(items)
    assert all(el.get("fill") == "none" for el in paths)
    assert all("DejaVu Sans" in (el.get("font-family") or "") for el in texts)
    assert any((el.get("id") or "").startswith("bg") for el in items)
    for inst in plan.instances:
        for k, e in enumerate(inst.elements):
            assert e.id == f"i{inst.index}_{k}"
    # окружности: либо A, либо четыре C
    d_bez = [el.get("d") or "" for el in paths if (el.get("d") or "").count(" C ") == 4]
    assert d_bez, "ожидаются окружности в виде четырех Безье"
    assert any("A" in (el.get("d") or "") for el in paths)


def test_legend_frames_are_clean() -> None:
    for seed in (0, 1):
        plan = _plan(seed)
        by_inst = {i.index: i for i in plan.instances}
        for q in plan.instances:
            if not q.is_query:
                continue
            own = {e.id for e in q.elements}
            assert not q.hard and not q.T.mirror and q.T.theta == 0.0
            assert q.T.s == pytest.approx(15.0 / plan.library[q.legend_id].diag)
            for e in plan.elements:
                if e.id in own:
                    continue
                assert not box_intersects(prim_bbox(e.prim), q.bbox), (seed, q.legend_id, e.id)
            label = plan.library[q.legend_id].label
            lab = [
                e.prim
                for e in plan.background
                if isinstance(e.prim, TextPrim) and e.prim.text == label
            ]
            assert len(lab) == 1
            assert lab[0].bbox[0] - q.bbox[2] >= 0.5 * 15.0
        assert by_inst  # индексы уникальны
        assert len(by_inst) == len(plan.instances)


def test_instances_do_not_overlap() -> None:
    inst = _plan(0).plan_instances
    for i, a in enumerate(inst):
        for b in inst[i + 1 :]:
            assert not box_intersects(a.bbox, b.bbox)


def test_export_files_and_coco(tmp_path: Path) -> None:
    plan = generate_plan(3, "small")
    paths = export_plan(plan, tmp_path, "s", raster=True)
    assert set(paths) == {"svg", "pdf", "gt", "coco", "png", "dirty_png", "dirty_coco"}
    gt = json.loads(paths["gt"].read_text())
    assert gt["units"] == "pt" and gt["width"] == 600 and gt["height"] == 400
    assert gt["similar_groups"][0] == ["duplex", "gfci", "r220"]
    assert len(gt["queries"]) == len(symbol_library())
    n_q = sum(1 for i in gt["instances"] if i["is_query"])
    assert n_q == len(gt["queries"])
    assert {"legend_id", "T", "bbox", "path_ids", "is_query", "hard"} <= set(gt["instances"][0])
    assert {"s", "theta", "tx", "ty", "mirror"} == set(gt["instances"][0]["T"])

    img = cv2.imread(str(paths["png"]), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (1667, 2500)
    assert (img < 128).sum() > 1000 and img.max() == 255
    dirty = cv2.imread(str(paths["dirty_png"]), cv2.IMREAD_GRAYSCALE)
    assert dirty.shape == img.shape and not np.array_equal(dirty, img)

    coco = json.loads(paths["coco"].read_text())
    assert coco["images"][0]["width"] == 2500 and coco["images"][0]["height"] == 1667
    assert coco["images"][0]["file_name"] == "s.png"
    assert [c["name"] for c in coco["categories"]] == ["legend_item", "object"]
    k = 300 / 72
    assert len(coco["annotations"]) == len(gt["instances"])
    for a, g in zip(coco["annotations"], gt["instances"], strict=True):
        x0, y0, x1, y1 = g["bbox"]
        want = [x0 * k, y0 * k, (x1 - x0) * k, (y1 - y0) * k]
        assert a["bbox"] == pytest.approx(want, abs=2e-3)
        assert a["category_id"] == (1 if g["is_query"] else 2)
        assert a["attributes"]["legend_id"] == g["legend_id"]
    legend_items = [a for a in coco["annotations"] if a["category_id"] == 1]
    assert len({a["attributes"]["legend_id"] for a in legend_items}) == len(legend_items)

    dcoco = json.loads(paths["dirty_coco"].read_text())
    assert dcoco["images"][0]["file_name"] == "s_dirty.png"
    shift = []
    for a, d in zip(coco["annotations"], dcoco["annotations"], strict=True):
        shift.append(np.abs(np.array(a["bbox"][:2]) - np.array(d["bbox"][:2])).max())
        assert d["bbox"][0] >= 0 and d["bbox"][1] >= 0
        assert d["bbox"][0] + d["bbox"][2] <= 2500 and d["bbox"][1] + d["bbox"][3] <= 1667
        assert d["bbox"][2] > 0.9 * a["bbox"][2]
    assert max(shift) > 0.1  # поворот учтен
    # определяется seed: повторный экспорт дает те же пиксели
    p2 = export_plan(plan, tmp_path / "again", "s", raster=True)
    assert p2["dirty_png"].read_bytes() == paths["dirty_png"].read_bytes()


def test_dirty_boxes_follow_rotation(tmp_path: Path) -> None:
    """Центр рамки легенды на грязном PNG остается на темных пикселях символа."""
    plan = generate_plan(4, "small")
    paths = export_plan(plan, tmp_path, "s", raster=True)
    dirty = cv2.imread(str(paths["dirty_png"]), cv2.IMREAD_GRAYSCALE)
    dcoco = json.loads(paths["dirty_coco"].read_text())
    for a in dcoco["annotations"]:
        if a["category_id"] != 1:
            continue
        x, y, w, h = (int(round(v)) for v in a["bbox"])
        crop = dirty[y : y + h, x : x + w]
        assert (crop < 100).sum() > 20


def test_pdf_opens(tmp_path: Path) -> None:
    plan = generate_plan(0, "small")
    paths = export_plan(plan, tmp_path, "s", raster=False)
    assert "png" not in paths and "dirty_png" not in paths
    assert not (tmp_path / "s.png").exists() and not (tmp_path / "s_dirty.png").exists()
    doc = pymupdf.open(paths["pdf"])
    page = doc[0]
    assert page.rect.width == pytest.approx(600.0, abs=0.01)
    assert page.rect.height == pytest.approx(400.0, abs=0.01)
    assert len(page.get_drawings()) > 0
    assert len(page.get_text("words")) > 0


def test_invalid_size_raises() -> None:
    with pytest.raises(ValueError):
        generate_plan(0, "huge")  # type: ignore[arg-type]


def test_generate_names_and_cli(tmp_path: Path) -> None:
    names = generate(2, "small", 7, tmp_path, raster=False)
    assert names == ["small_000", "small_001"]
    gt = json.loads((tmp_path / "small_001_gt.json").read_text())
    assert gt["name"] == "small_001"
    assert (tmp_path / "small_001.svg").read_text() == plan_to_svg(generate_plan(7001, "small"))
    out = tmp_path / "cli"
    cli_main(["--n", "1", "--size", "small", "--seed", "0", "--out", str(out), "--no-raster"])
    assert (out / "small_000.svg").exists() and not (out / "small_000.png").exists()


@pytest.mark.slow
def test_large_primitive_count(tmp_path: Path) -> None:
    plan = generate_plan(0, "large")
    assert 21000 <= plan.n_prims <= 39000
    assert 400 <= len(plan.plan_instances) <= 800
    assert plan.width == 1728 and plan.height == 1152
    paths = export_plan(plan, tmp_path, "l", raster=False)
    doc = pymupdf.open(paths["pdf"])
    assert doc[0].rect.width == pytest.approx(1728.0, abs=0.01)
