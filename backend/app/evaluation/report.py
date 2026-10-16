"""Вывод результатов оценки: таблица в Markdown, PR-кривая в CSV и SVG."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .metrics import PRPoint, Report, Scores


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(report: Report, symbol_names: dict[str, str] | None = None) -> str:
    names = symbol_names or {}
    lines = [
        "# Оценка качества поиска",
        "",
        (
            f"Алгоритм: `{report.detector or 'не указан'}` · порог IoU: {report.iou_threshold} · "
            f"порог уверенности: {report.score_threshold} · таблица обозначений: "
            f"{'учитывается' if report.include_legend else 'не учитывается'}"
        ),
        "",
        "## Итог",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Precision (micro) | {fmt(report.overall.precision)} |",
        f"| Recall (micro) | {fmt(report.overall.recall)} |",
        f"| F1 (micro) | {fmt(report.overall.f1)} |",
        f"| Средний IoU попаданий | {fmt(report.overall.mean_iou)} |",
        f"| AP (все обозначения вместе) | {fmt(report.overall.ap)} |",
        (
            f"| Precision / Recall / F1 (macro) | {fmt(report.macro['precision'])} / "
            f"{fmt(report.macro['recall'])} / {fmt(report.macro['f1'])} |"
        ),
        f"| mAP (среднее AP по обозначениям) | {fmt(report.macro['map'])} |",
        f"| TP / FP / FN | {report.overall.tp} / {report.overall.fp} / {report.overall.fn} |",
        "",
        "## По обозначениям",
        "",
        "| Обозначение | Эталонов | TP | FP | FN | Precision | Recall | F1 | IoU | AP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_row(s, names.get(s.symbol, s.symbol)) for s in report.per_symbol]
    if report.timing:
        t = report.timing
        lines += [
            "",
            "## Время",
            "",
            f"Чертежей: {t.files}, листов: {t.pages}, всего {t.total_s:.3f} с.",
            "",
            "| Чертёж | Время, с |",
            "|---|---|",
            *(f"| {escape(f)} | {s:.3f} |" for f, s in t.per_file_s.items()),
            "",
            (
                f"Среднее на чертёж: {t.mean_per_file_s:.3f} с · медиана: {t.median_per_file_s:.3f} с · "
                f"максимум: {t.max_per_file_s:.3f} с · среднее на лист: {t.mean_per_page_s:.3f} с."
            ),
        ]
    fn = sum(1 for e in report.errors if e["type"] == "fn")
    fp = len(report.errors) - fn
    lines += [
        "",
        "## Ошибки",
        "",
        (
            f"Пропусков: {fn}, ложных срабатываний: {fp}. Полный список — в `metrics.json`, "
            "раздел `errors`."
        ),
    ]
    if report.ignored_predictions:
        lines.append(
            f"Ответов в таблице условных обозначений (не оценивались): {report.ignored_predictions}."
        )
    return "\n".join(lines) + "\n"


def _row(s: Scores, name: str) -> str:
    return (
        f"| {escape(name)} | {s.n_truth} | {s.tp} | {s.fp} | {s.fn} | {fmt(s.precision)} | "
        f"{fmt(s.recall)} | {fmt(s.f1)} | {fmt(s.mean_iou)} | {fmt(s.ap)} |"
    )


def pr_csv(points: list[PRPoint]) -> str:
    rows = ["threshold,precision,recall"]
    rows += [f"{p.threshold:.6f},{p.precision:.6f},{p.recall:.6f}" for p in points]
    return "\n".join(rows) + "\n"


def pr_svg(points: list[PRPoint], title: str = "PR-кривая") -> str:
    """График precision от recall. Ступенчатая линия, как принято для PR-кривых."""
    w, h, left, top, size = 420, 400, 60, 40, 320

    def sx(recall: float) -> float:
        return left + recall * size

    def sy(precision: float) -> float:
        return top + (1 - precision) * size

    grid = []
    for k in range(6):
        v = k / 5
        grid.append(
            f'<line x1="{sx(0)}" y1="{sy(v)}" x2="{sx(1)}" y2="{sy(v)}" stroke="#ddd"/>'
            f'<line x1="{sx(v)}" y1="{sy(0)}" x2="{sx(v)}" y2="{sy(1)}" stroke="#ddd"/>'
            f'<text x="{left - 8}" y="{sy(v) + 4}" text-anchor="end">{v:.1f}</text>'
            f'<text x="{sx(v)}" y="{sy(0) + 18}" text-anchor="middle">{v:.1f}</text>'
        )
    path = ""
    if points:
        coords = [f"M{sx(0):.1f},{sy(points[0].precision):.1f}"]
        for p in points:
            coords.append(f"H{sx(p.recall):.1f}V{sy(p.precision):.1f}")
        path = (
            f'<path d="{" ".join(coords)}" fill="none" stroke="#1f6feb" stroke-width="2"/>'
            + "".join(
                f'<circle cx="{sx(p.recall):.1f}" cy="{sy(p.precision):.1f}" r="2.5" '
                f'fill="#1f6feb"><title>порог {p.threshold:.3f}: P={p.precision:.3f}, '
                f"R={p.recall:.3f}</title></circle>"
                for p in points
            )
        )
    else:
        path = (
            f'<text x="{sx(0.5)}" y="{sy(0.5)}" text-anchor="middle" fill="#888">нет ответов</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12" fill="#333">'
        f'<rect width="{w}" height="{h}" fill="#fff"/>'
        f'<text x="{w / 2}" y="22" text-anchor="middle" font-size="14">{escape(title)}</text>'
        + "".join(grid)
        + f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="#999"/>'
        + path
        + f'<text x="{sx(0.5)}" y="{h - 6}" text-anchor="middle">Recall</text>'
        f'<text x="16" y="{sy(0.5)}" text-anchor="middle" '
        f'transform="rotate(-90 16 {sy(0.5)})">Precision</text>'
        "</svg>\n"
    )


def write_all(report: Report, out_dir: Path, symbol_names: dict[str, str] | None = None) -> None:
    """Сохраняет metrics.json, summary.md, pr_curve.csv и pr_curve.svg в out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(markdown(report, symbol_names), encoding="utf-8")
    (out_dir / "pr_curve.csv").write_text(pr_csv(report.pr_curve), encoding="utf-8")
    title = f"PR-кривая · AP = {fmt(report.overall.ap)}"
    (out_dir / "pr_curve.svg").write_text(pr_svg(report.pr_curve, title), encoding="utf-8")
