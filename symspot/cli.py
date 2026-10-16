"""CLI (typer): run, eval, synth. Фаза 4."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from symspot.core.types import Result

app = typer.Typer(add_completion=False, help="SymSpot: поиск обозначений на чертежах")
console = Console()
err_console = Console(stderr=True)

ALL_FORMATS = ("svg", "pdf", "png", "html")
FORMAT_FILES = {
    "svg": "overlay.svg",
    "pdf": "overlay.pdf",
    "png": "overlay.png",
    "html": "report.html",
}
TOP_CONFUSIONS = 10
EXIT_ERROR = 1


class CliError(Exception):
    """Ожидаемая ошибка пользователя: печатается без трейсбека."""


def _fail(e: Exception, verbose: bool) -> typer.Exit:
    if verbose:
        err_console.print_exception()
    else:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(e))}", highlight=False)
    return typer.Exit(EXIT_ERROR)


def _parse_sets(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise CliError(f"--set ожидает key=value, получено {item!r}")
        try:
            out[key] = yaml.safe_load(value)
        except yaml.YAMLError as e:
            raise CliError(f"--set {item!r}: не удалось разобрать значение ({e})") from e
    return out


def _parse_formats(text: str) -> list[str]:
    fmts = [f.strip().lower() for f in text.split(",") if f.strip()]
    bad = [f for f in fmts if f not in ALL_FORMATS]
    if bad:
        raise CliError(f"неизвестные форматы {bad}; допустимы {', '.join(ALL_FORMATS)}")
    return list(dict.fromkeys(fmts))


def _check_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise CliError(f"{what} не найден: {path}")


def _compute_metrics(result: Result, gt_path: Path, iou: float) -> dict:
    from symspot.eval.metrics import evaluate
    from symspot.io.annotations import load_ground_truth, load_similar_groups

    gt = load_ground_truth(
        gt_path, drawing_size=(result.width, result.height), image_name=result.drawing
    )
    groups = load_similar_groups(gt_path)
    metrics = evaluate(result.instances, gt, iou_thr=iou, similar_groups=groups)
    return metrics.to_dict()


def _print_metrics(m: dict) -> None:
    table = Table(title="Метрики по классам")
    table.add_column("класс")
    for col in ("P", "R", "F1", "tp", "fp", "fn"):
        table.add_column(col, justify="right")
    for cls, v in m["per_class"].items():
        table.add_row(
            cls,
            f"{v['precision']:.3f}",
            f"{v['recall']:.3f}",
            f"{v['f1']:.3f}",
            str(v["tp"]),
            str(v["fp"]),
            str(v["fn"]),
        )
    o = m["overall"]
    table.add_section()
    table.add_row(
        "ИТОГО",
        f"{o['precision']:.3f}",
        f"{o['recall']:.3f}",
        f"{o['f1']:.3f}",
        str(o["tp"]),
        str(o["fp"]),
        str(o["fn"]),
    )
    console.print(table)
    console.print(
        f"similar_group_error: {m['similar_group_error']:.3f} "
        f"(символов в группах: {m['n_similar']})"
    )

    cells = [
        (g, p, n) for g, row in m["confusion"].items() for p, n in row.items() if g != p and n > 0
    ]
    cells.sort(key=lambda c: (-c[2], c[0], c[1]))
    if cells:
        conf = Table(title=f"Confusion вне диагонали (топ {TOP_CONFUSIONS})")
        conf.add_column("GT")
        conf.add_column("предсказано")
        conf.add_column("число", justify="right")
        for g, p, n in cells[:TOP_CONFUSIONS]:
            conf.add_row(g, p, str(n))
        console.print(conf)


def _print_summary(result: Result, files: list[Path]) -> None:
    found = [i for i in result.instances if not i.is_query]
    n_query = len(result.instances) - len(found)
    counts = Counter(i.legend_id for i in found)
    amb = Counter(i.legend_id for i in found if i.status == "ambiguous")

    table = Table(
        title=f"{Path(result.drawing).name}: найдено {len(found)} (эталонов среди них: {n_query})"
    )
    table.add_column("класс")
    table.add_column("экземпляров", justify="right")
    table.add_column("ambiguous", justify="right")
    for cls in sorted(counts):
        table.add_row(cls, str(counts[cls]), str(amb.get(cls, 0)))
    table.add_section()
    table.add_row("ВСЕГО", str(len(found)), str(sum(amb.values())))
    console.print(table)

    t = Table(title="Тайминги, с")
    t.add_column("этап")
    t.add_column("время", justify="right")
    for name, sec in result.timings.items():
        t.add_row(name, f"{sec:.2f}")
    console.print(t)

    for f in files:
        console.print(f"  {f}", highlight=False, markup=False)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


@app.command()
def run(
    drawing: Annotated[Path, typer.Argument(help="чертеж: SVG, PDF или растр")],
    queries: Annotated[Path, typer.Option("--queries", help="эталоны: COCO или простой JSON")],
    out: Annotated[Path, typer.Option("--out", help="каталог результата")],
    config: Annotated[Path | None, typer.Option("--config", help="YAML с порогами")] = None,
    page: Annotated[int, typer.Option("--page", help="номер страницы PDF")] = 0,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="переопределение поля конфига key=value (повторяемое)"),
    ] = None,
    formats: Annotated[
        str, typer.Option("--formats", help="через запятую: svg,pdf,png,html")
    ] = ",".join(ALL_FORMATS),
    gt: Annotated[Path | None, typer.Option("--gt", help="разметка для метрик")] = None,
    iou: Annotated[float, typer.Option("--iou", help="порог IoU для метрик")] = 0.5,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="DEBUG-лог и трейсбеки")] = False,
) -> None:
    """Найти обозначения на чертеже по эталонам и сохранить раскраску."""
    if verbose:
        logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
        logging.getLogger("symspot").setLevel(logging.DEBUG)
    try:
        fmts = _parse_formats(formats)
        overrides = _parse_sets(set_ or [])
        _check_file(drawing, "чертеж")
        _check_file(queries, "файл эталонов")
        if config is not None:
            _check_file(config, "конфиг")
        if gt is not None:
            _check_file(gt, "файл разметки")

        from symspot.core.config import load_config
        from symspot.pipeline import run as run_pipeline

        try:
            cfg = load_config(config, overrides)
        except ValueError as e:  # pydantic.ValidationError наследуется от ValueError
            raise CliError(f"конфиг: {e}") from e

        result, norm = run_pipeline(drawing, queries, cfg, page=page)

        out.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        result_path = out / "result.json"
        _write_json(result_path, result.to_dict())
        files.append(result_path)

        if fmts:
            from symspot.render.html_report import render_html
            from symspot.render.pdf_overlay import render_pdf
            from symspot.render.png_overlay import render_png
            from symspot.render.svg_overlay import render_svg

            renderers = {
                "svg": render_svg,
                "pdf": render_pdf,
                "png": render_png,
                "html": render_html,
            }
            for fmt in fmts:
                files.append(Path(renderers[fmt](norm, result, out / FORMAT_FILES[fmt])))

        metrics: dict | None = None
        if gt is not None:
            metrics = _compute_metrics(result, gt, iou)
            metrics_path = out / "metrics.json"
            _write_json(metrics_path, metrics)
            files.append(metrics_path)

        _print_summary(result, files)
        if metrics is not None:
            _print_metrics(metrics)
    except typer.Exit:
        raise
    except Exception as e:  # noqa: BLE001 - любая ошибка пайплайна -> понятное сообщение
        raise _fail(e, verbose) from None


@app.command("eval")
def eval_(
    result_json: Annotated[Path, typer.Argument(help="result.json, записанный командой run")],
    gt: Annotated[Path, typer.Argument(help="разметка экземпляров: COCO или GT синтетики")],
    iou: Annotated[float, typer.Option("--iou", help="порог IoU")] = 0.5,
    out: Annotated[Path | None, typer.Option("--out", help="куда записать metrics.json")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="трейсбеки")] = False,
) -> None:
    """Посчитать метрики результата относительно разметки."""
    try:
        _check_file(result_json, "результат")
        _check_file(gt, "файл разметки")
        try:
            data = json.loads(result_json.read_text(encoding="utf-8"))
            result = Result.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise CliError(f"{result_json}: не result.json SymSpot ({e!r})") from e
        metrics = _compute_metrics(result, gt, iou)
        _print_metrics(metrics)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            _write_json(out, metrics)
            console.print(f"  {out}", highlight=False, markup=False)
    except typer.Exit:
        raise
    except Exception as e:  # noqa: BLE001
        raise _fail(e, verbose) from None


@app.command()
def synth(
    n: Annotated[int, typer.Option("--n", help="число чертежей")] = 20,
    size: Annotated[str, typer.Option("--size", help="small или large")] = "small",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    out: Annotated[Path, typer.Option("--out", help="каталог")] = Path("data/synth"),
    no_raster: Annotated[bool, typer.Option("--no-raster", help="не создавать PNG")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="трейсбеки")] = False,
) -> None:
    """Сгенерировать синтетические чертежи с разметкой."""
    try:
        if size not in ("small", "large"):
            raise CliError(f"--size: ожидалось small или large, получено {size!r}")
        if n < 1:
            raise CliError("--n должно быть не меньше 1")
        from symspot.synth.export import generate

        names = generate(n, size, seed, out, raster=not no_raster)
        console.print(f"{len(names)} чертежей в {out}", highlight=False, markup=False)
    except typer.Exit:
        raise
    except Exception as e:  # noqa: BLE001
        raise _fail(e, verbose) from None


if __name__ == "__main__":
    app()
