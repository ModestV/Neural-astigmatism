"""Тесты CLI (typer CliRunner) на синтетике small."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from symspot.cli import app
from symspot.synth.export import generate

runner = CliRunner()
NAME = "small_000"


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("synth")
    generate(1, "small", 0, d, raster=False)
    return d


def _run(synth_dir: Path, out: Path, ext: str, *extra: str):
    gt = synth_dir / f"{NAME}_gt.json"
    return runner.invoke(
        app,
        [
            "run",
            str(synth_dir / f"{NAME}.{ext}"),
            "--queries",
            str(gt),
            "--out",
            str(out),
            "--gt",
            str(gt),
            *extra,
        ],
    )


@pytest.fixture(scope="module")
def svg_out(synth_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("out_svg")
    res = _run(synth_dir, out, "svg")
    assert res.exit_code == 0, res.output
    return out


def test_run_svg_writes_all_files(svg_out: Path) -> None:
    for name in (
        "result.json",
        "overlay.svg",
        "overlay.pdf",
        "overlay.png",
        "report.html",
        "metrics.json",
    ):
        assert (svg_out / name).stat().st_size > 0, name
    data = json.loads((svg_out / "result.json").read_text(encoding="utf-8"))
    assert data["kind"] == "vector"
    assert len(data["instances"]) > 0
    assert "total" in data["timings"]


def test_run_svg_metrics(svg_out: Path) -> None:
    m = json.loads((svg_out / "metrics.json").read_text(encoding="utf-8"))
    assert m["overall"]["f1"] >= 0.9


def test_run_pdf(synth_dir: Path, tmp_path: Path) -> None:
    res = _run(synth_dir, tmp_path / "o", "pdf", "--formats", "svg")
    assert res.exit_code == 0, res.output
    m = json.loads((tmp_path / "o" / "metrics.json").read_text(encoding="utf-8"))
    assert m["overall"]["f1"] >= 0.9


def test_run_set_overrides_config(synth_dir: Path, tmp_path: Path) -> None:
    res = _run(
        synth_dir,
        tmp_path,
        "svg",
        "--formats",
        "",
        "--set",
        "accept_score=0.99",
        "--set",
        "allow_mirror=false",
    )
    assert res.exit_code == 0, res.output
    cfg = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["config"]
    assert cfg["accept_score"] == 0.99
    assert cfg["allow_mirror"] is False


def test_run_formats_svg_only(synth_dir: Path, tmp_path: Path) -> None:
    res = _run(synth_dir, tmp_path, "svg", "--formats", "svg")
    assert res.exit_code == 0, res.output
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"result.json", "overlay.svg", "metrics.json"}


def test_run_bad_set_key_exit_1(synth_dir: Path, tmp_path: Path) -> None:
    res = _run(synth_dir, tmp_path, "svg", "--set", "no_such_field=1")
    assert res.exit_code == 1
    assert "Traceback" not in res.output


def test_run_bad_set_syntax_exit_1(synth_dir: Path, tmp_path: Path) -> None:
    res = _run(synth_dir, tmp_path, "svg", "--set", "accept_score")
    assert res.exit_code == 1
    assert "key=value" in res.output


def test_run_bad_format_exit_1(synth_dir: Path, tmp_path: Path) -> None:
    res = _run(synth_dir, tmp_path, "svg", "--formats", "gif")
    assert res.exit_code == 1


def test_run_missing_drawing_exit_1(synth_dir: Path, tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "run",
            str(tmp_path / "nope.svg"),
            "--queries",
            str(synth_dir / f"{NAME}_gt.json"),
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert res.exit_code == 1
    assert "Ошибка" in res.output
    assert "Traceback" not in res.output


def test_run_no_queries_exit_1(synth_dir: Path, tmp_path: Path) -> None:
    empty = tmp_path / "q.json"
    empty.write_text("[]", encoding="utf-8")
    res = runner.invoke(
        app,
        [
            "run",
            str(synth_dir / f"{NAME}.svg"),
            "--queries",
            str(empty),
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert res.exit_code == 1
    assert "Traceback" not in res.output


def test_eval_on_result(svg_out: Path, synth_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    res = runner.invoke(
        app,
        [
            "eval",
            str(svg_out / "result.json"),
            str(synth_dir / f"{NAME}_gt.json"),
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    m = json.loads(out.read_text(encoding="utf-8"))
    assert m["overall"]["f1"] >= 0.9
    assert m == json.loads((svg_out / "metrics.json").read_text(encoding="utf-8"))
    assert "ИТОГО" in res.output


def test_eval_missing_file_exit_1(synth_dir: Path, tmp_path: Path) -> None:
    res = runner.invoke(
        app, ["eval", str(tmp_path / "none.json"), str(synth_dir / f"{NAME}_gt.json")]
    )
    assert res.exit_code == 1


def test_eval_not_result_exit_1(synth_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "r.json"
    bad.write_text("{}", encoding="utf-8")
    res = runner.invoke(app, ["eval", str(bad), str(synth_dir / f"{NAME}_gt.json")])
    assert res.exit_code == 1
    assert "Traceback" not in res.output


def test_synth_creates_files(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "synth",
            "--n",
            "1",
            "--size",
            "small",
            "--seed",
            "1",
            "--out",
            str(tmp_path),
            "--no-raster",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "small_000.svg").is_file()
    assert (tmp_path / "small_000_gt.json").is_file()
    assert not list(tmp_path.glob("*.png"))


def test_synth_bad_size_exit_1(tmp_path: Path) -> None:
    res = runner.invoke(app, ["synth", "--size", "huge", "--out", str(tmp_path)])
    assert res.exit_code == 1
