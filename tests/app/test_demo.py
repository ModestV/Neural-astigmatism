"""Тесты демо: логика без Streamlit и smoke через AppTest."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "app" / "demo.py"
SMOKE_TIMEOUT = 180


def _load_demo():
    spec = importlib.util.spec_from_file_location("symspot_app_demo", DEMO)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def demo():
    return _load_demo()


@pytest.fixture(scope="module")
def example(demo, tmp_path_factory):
    d = tmp_path_factory.mktemp("example")
    svg, gt = demo.make_example(0, d)
    return svg, gt


@pytest.fixture(scope="module")
def out(demo, example, tmp_path_factory):
    svg, gt = example
    work = tmp_path_factory.mktemp("work")
    return demo.run_search(svg, gt, demo.build_config(), work_dir=work)


def test_run_returns_result_and_table(demo, out):
    assert out.result.instances
    assert len(out.table) == len(out.result.instances)
    assert list(out.table.columns) == demo.TABLE_COLUMNS
    assert out.table["id"].is_unique
    assert out.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert "total" in out.result.timings


def test_metrics_computed(demo, out):
    assert out.metrics is not None
    mt = demo.metrics_table(out.metrics)
    assert mt["класс"].iloc[-1] == "ИТОГО"
    assert {"precision", "recall", "f1", "tp", "fp", "fn"} <= set(mt.columns)
    assert 0.0 <= out.metrics["overall"]["f1"] <= 1.0


def test_metrics_absent_for_plain_queries(demo, out, tmp_path):
    plain = tmp_path / "queries.json"
    q = out.result.queries[0]
    plain.write_text(
        json.dumps({"legend_items": [{"legend_id": q.legend_id, "bbox": list(q.bbox)}]}),
        encoding="utf-8",
    )
    metrics, note = demo.try_metrics(out.result, plain)
    assert metrics is None
    assert note


def test_downloads_valid(demo, out):
    d = out.downloads
    assert all(d[k] for k in ("svg", "pdf", "json", "html"))
    assert not out.download_errors
    assert json.loads(d["json"])["instances"]
    assert d["svg"].lstrip().startswith((b"<?xml", b"<"))
    assert d["pdf"].startswith(b"%PDF")
    assert b"<html" in d["html"].lower()


def test_filter_and_ambiguous(demo, out):
    cls = out.table["legend_id"].iloc[0]
    only = demo.filter_table(out.table, [cls])
    assert set(only["legend_id"]) == {cls}
    assert demo.filter_table(out.table, None).shape == out.table.shape
    no_q = demo.filter_table(out.table, show_queries=False)
    assert not no_q["is_query"].any()
    assert (demo.ambiguous_table(out.table)["status"] == "ambiguous").all()
    summary = demo.class_summary(out.table)
    assert summary["found"].sum() == len(no_q)


def test_save_upload_keeps_name(demo, tmp_path):
    p = demo.save_upload("../evil/sheet 1.svg", b"<svg/>", tmp_path / "up")
    assert p.name == "sheet 1.svg"
    assert p.parent == tmp_path / "up"
    assert p.read_bytes() == b"<svg/>"
    with pytest.raises(ValueError):
        demo.save_upload("", b"x", tmp_path)


def test_build_config_overrides(demo):
    cfg = demo.build_config(accept_score=0.9, allow_mirror=False)
    assert cfg.accept_score == 0.9
    assert cfg.allow_mirror is False
    assert cfg.cover_tol == demo.Config().cover_tol


def test_apptest_smoke():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO), default_timeout=SMOKE_TIMEOUT).run()
    assert not at.exception


def test_apptest_example_and_run():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO), default_timeout=SMOKE_TIMEOUT).run()
    at.sidebar.button(key="btn_example").click().run()
    assert not at.exception
    at.sidebar.button(key="btn_run").click().run()
    assert not at.exception
    assert not at.error
    assert at.session_state["out"].result.instances
    # повторная отрисовка не перезапускает поиск
    first = at.session_state["out"]
    at.run()
    assert at.session_state["out"] is first
