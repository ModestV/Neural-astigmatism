"""Фаза 4: векторный пайплайн целиком на синтетике small (seed 0)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from symspot.core.config import Config
from symspot.core.geometry_utils import bbox_diag, bbox_expand, bbox_iou, prim_bbox
from symspot.core.types import Result
from symspot.eval.metrics import evaluate
from symspot.io.annotations import load_ground_truth, load_similar_groups
from symspot.pipeline import run

from .conftest import Run

F1_MIN = 0.92
SIMILAR_ERR_MAX = 0.05
SECONDS_MAX = 5.0


def _metrics(r: Run):
    gt = load_ground_truth(
        r.gt_path, drawing_size=(r.drawing.width, r.drawing.height), image_name=r.name
    )
    return evaluate(r.result.instances, gt, similar_groups=load_similar_groups(r.gt_path))


def _mean_f1(runs: list[Run], ext: str) -> tuple[float, list[float]]:
    f1s = [_metrics(r).overall.f1 for r in runs if r.ext == ext]
    return float(np.mean(f1s)), f1s


@pytest.mark.slow
@pytest.mark.parametrize("ext", ["svg", "pdf"])
def test_mean_f1_on_small_synth(runs, ext):
    mean, f1s = _mean_f1(runs, ext)
    assert len(f1s) == 3
    assert mean >= F1_MIN, f"{ext}: средний F1 {mean:.3f}, по чертежам {f1s}"


@pytest.mark.slow
@pytest.mark.parametrize("ext", ["svg", "pdf"])
def test_similar_group_error_on_small_synth(runs, ext):
    errs = [(_metrics(r).similar_group_error, _metrics(r).n_similar) for r in runs if r.ext == ext]
    assert all(n > 0 for _, n in errs), "в GT должны быть символы из групп похожих"
    total_n = sum(n for _, n in errs)
    mean_err = sum(e * n for e, n in errs) / total_n
    assert mean_err <= SIMILAR_ERR_MAX, f"{ext}: ошибка внутри групп {mean_err:.3f}, {errs}"


@pytest.mark.slow
def test_each_drawing_has_reasonable_precision_and_recall(runs):
    for r in runs:
        m = _metrics(r).overall
        assert m.precision >= 0.85, (r.name, r.ext, m.to_dict())
        assert m.recall >= 0.85, (r.name, r.ext, m.to_dict())


@pytest.mark.slow
def test_time_per_small_drawing_under_5_seconds(runs):
    slow = [(r.name, r.ext, round(r.seconds, 2)) for r in runs if r.seconds >= SECONDS_MAX]
    assert not slow, f"прогон small дольше {SECONDS_MAX} с: {slow}"


@pytest.mark.slow
def test_wall_through_instances_only_get_pieces_inside_gt_box(runs):
    found_total = 0
    for r in runs:
        raw = json.loads(r.gt_path.read_text(encoding="utf-8"))
        gt = load_ground_truth(
            r.gt_path, drawing_size=(r.drawing.width, r.drawing.height), image_name=r.name
        )
        raw_inst = raw["instances"]
        assert len(raw_inst) == len(gt), "порядок GT должен совпадать с файлом"
        by_id = r.drawing.prim_by_id()
        for g, rg in zip(gt, raw_inst, strict=True):
            if "wall_through" not in rg.get("hard", []):
                continue
            cands = [
                (bbox_iou(i.bbox, g.bbox), i)
                for i in r.result.instances
                if i.legend_id == g.legend_id
            ]
            if not cands:
                continue
            iou, inst = max(cands, key=lambda t: t[0])
            if iou < 0.5:
                continue
            found_total += 1
            box = bbox_expand(g.bbox, 0.10 * bbox_diag(g.bbox))
            for pid in inst.piece_ids:
                b = prim_bbox(by_id[pid])
                inside = box[0] <= b[0] and box[1] <= b[1] and b[2] <= box[2] and b[3] <= box[3]
                assert inside, (
                    f"{r.name}.{r.ext}: {g.legend_id}: кусок {pid} {b} вне рамки GT {box} "
                    "(окрашена стена?)"
                )
    assert found_total >= 5, "слишком мало найденных экземпляров с wall_through для проверки"


@pytest.mark.slow
def test_wall_through_instances_are_mostly_found(runs):
    total = found = 0
    for r in runs:
        raw = json.loads(r.gt_path.read_text(encoding="utf-8"))
        gt = load_ground_truth(
            r.gt_path, drawing_size=(r.drawing.width, r.drawing.height), image_name=r.name
        )
        for g, rg in zip(gt, raw["instances"], strict=True):
            if "wall_through" in rg.get("hard", []):
                total += 1
                found += any(
                    i.legend_id == g.legend_id and bbox_iou(i.bbox, g.bbox) >= 0.5
                    for i in r.result.instances
                )
    assert total > 0
    assert found / total >= 0.85, f"найдено {found} из {total}"


@pytest.mark.slow
def test_instances_are_well_formed(runs):
    for r in runs:
        ids = [i.id for i in r.result.instances]
        assert ids == sorted(set(ids)), "id экземпляров уникальны"
        prim_ids = set(r.drawing.prim_by_id())
        seen: dict[int, int] = {}
        for i in r.result.instances:
            assert i.status in ("ok", "ambiguous")
            assert 0.0 <= i.score <= 1.0 + 1e-9
            assert i.bbox[0] <= i.bbox[2] and i.bbox[1] <= i.bbox[3]
            assert set(i.piece_ids) <= prim_ids
            for p in i.piece_ids:
                seen[p] = seen.get(p, 0) + 1
        # один кусок принадлежит не более чем одному экземпляру
        assert max(seen.values(), default=0) == 1, f"{r.name}.{r.ext}: куски заняты дважды"


# --- быстрые проверки на одном чертеже ---


def test_result_to_dict_is_json_serializable(one_run):
    d = one_run.result.to_dict()
    text = json.dumps(d)
    back = json.loads(text)
    assert back["drawing"] == d["drawing"]
    assert len(back["instances"]) == len(one_run.result.instances) > 0
    assert set(back["timings"]) == set(d["timings"])


def test_result_roundtrip_returns_same_instances(one_run):
    res = one_run.result
    again = Result.from_dict(json.loads(json.dumps(res.to_dict())))
    assert len(again.instances) == len(res.instances)
    for a, b in zip(res.instances, again.instances, strict=True):
        assert a.to_dict() == b.to_dict()
        assert a == b
    assert again.drawing == res.drawing
    assert [q.legend_id for q in again.queries] == [q.legend_id for q in res.queries]
    assert again.kind == res.kind and again.width == res.width


def test_result_from_to_dict_without_json(one_run):
    res = one_run.result
    again = Result.from_dict(res.to_dict())
    assert again.instances == res.instances


def test_timings_present_and_non_negative(one_run):
    t = one_run.result.timings
    assert t and all(v >= 0 for v in t.values())


def test_no_queries_raises_value_error(tmp_path, synth_dir, synth_names):
    drawing = synth_dir / f"{synth_names[0]}.svg"
    empty_list = tmp_path / "empty.json"
    empty_list.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        run(drawing, empty_list, Config())
    raw = json.loads((synth_dir / f"{synth_names[0]}_gt.json").read_text(encoding="utf-8"))
    raw["queries"] = []
    no_q = tmp_path / "no_queries_gt.json"
    no_q.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        run(drawing, no_q, Config())


def test_pipeline_is_deterministic(synth_dir, synth_names, one_run):
    t0 = time.perf_counter()
    again, _ = run(
        synth_dir / f"{synth_names[0]}.svg", synth_dir / f"{synth_names[0]}_gt.json", Config()
    )
    assert time.perf_counter() - t0 < 30
    a = [i.to_dict() for i in one_run.result.instances]
    b = [i.to_dict() for i in again.instances]
    assert a == b


def test_pipeline_returns_normalized_drawing(one_run):
    d = one_run.drawing
    assert d.kind == "vector" and d.normalized
    assert len(d.primitives) > 100
    assert isinstance(Path(one_run.result.drawing).name, str)
