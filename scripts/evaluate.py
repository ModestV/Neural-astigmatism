"""Оценка качества поиска условных обозначений по размеченным чертежам.

    backend/.venv/bin/python scripts/evaluate.py                          # заглушка на samples/
    backend/.venv/bin/python scripts/evaluate.py --detector my            # свой алгоритм
    backend/.venv/bin/python scripts/evaluate.py --pred answers.json      # готовые ответы
    backend/.venv/bin/python scripts/evaluate.py --simulate               # проверка самих метрик

Разметка (--gt) и ответы (--pred) — JSON в формате samples/ground_truth.json,
подробности в backend/app/evaluation/dataset.py.

Результат — в report/results/<алгоритм>/:
  predictions.json   ответы алгоритма (если он запускался)
  metrics.json       все метрики, PR-кривая и список ошибок
  summary.md         сводка для отчёта
  pr_curve.csv/.svg  PR-кривая
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.detection import DETECTORS, get_detector
from app.evaluation import Annotations, Instance, evaluate, load, save
from app.evaluation.dataset import DatasetFormatError
from app.evaluation.report import fmt, write_all
from app.evaluation.run import RunError, run_detector
from app.models import SearchParams


def main() -> int:
    args = parse_args()
    try:
        truth = load(args.gt)
        if args.pred:
            predictions = load(args.pred)
            name = predictions.detector or args.pred.stem
        elif args.simulate:
            predictions = simulate(truth, random.Random(args.seed))
            name = "simulated"
        else:
            detector = get_detector(args.detector)
            print(f"Алгоритм «{detector.name}»: {detector.description}")
            params = SearchParams(threshold=args.score_threshold or 0.8)
            predictions = run_detector(
                detector, truth, args.drawings or args.gt.parent, params, progress=print
            )
            name = detector.name
    except (DatasetFormatError, RunError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    out = args.out or ROOT / "report" / "results" / name
    report = evaluate(truth, predictions, args.iou, args.score_threshold, args.include_legend)
    if not args.pred:
        out.mkdir(parents=True, exist_ok=True)
        save(predictions, out / "predictions.json")
    write_all(report, out, truth.symbol_names)
    print_summary(report)
    print(f"\nРезультаты: {out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Метрики качества поиска условных обозначений")
    parser.add_argument(
        "--gt", type=Path, default=ROOT / "samples" / "ground_truth.json", help="разметка"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--detector",
        default="stub",
        help=f"какой алгоритм запустить: {', '.join(DETECTORS)} (по умолчанию stub)",
    )
    source.add_argument(
        "--pred", type=Path, help="готовые ответы алгоритма — алгоритм не запускается"
    )
    source.add_argument(
        "--simulate",
        action="store_true",
        help="вместо алгоритма — разметка с шумом, пропусками и ложными срабатываниями",
    )
    parser.add_argument(
        "--drawings", type=Path, help="папка с чертежами (по умолчанию — папка разметки)"
    )
    parser.add_argument("--out", type=Path, help="куда сохранить результаты")
    parser.add_argument("--iou", type=float, default=0.5, help="порог IoU для попадания (0.5)")
    parser.add_argument(
        "--score-threshold", type=float, default=0.0, help="отбросить ответы с меньшей уверенностью"
    )
    parser.add_argument(
        "--include-legend", action="store_true", help="оценивать и таблицу условных обозначений"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed для --simulate")
    return parser.parse_args()


def simulate(truth: Annotations, rng: random.Random) -> Annotations:
    """Правдоподобные ответы «алгоритма»: 85 % экземпляров найдены со сдвигом рамки,
    плюс по 3 ложных срабатывания на лист. Нужны, чтобы проверить метрики и графики,
    пока настоящего алгоритма нет."""
    result = Annotations(symbol_names=dict(truth.symbol_names), detector="simulated")
    symbols = sorted({i.symbol for i in truth.instances})
    for file, page in truth.pages():
        on_page = [i for i in truth.instances if i.file == file and i.page == page]
        for inst in on_page:
            if rng.random() > 0.85:
                continue
            x, y, w, h = inst.bbox
            k = rng.uniform(0.9, 1.1)
            bbox = (
                x + rng.uniform(-0.15, 0.15) * w,
                y + rng.uniform(-0.15, 0.15) * h,
                w * k,
                h * k,
            )
            result.instances.append(
                Instance(file, page, inst.symbol, bbox, score=rng.uniform(0.5, 1.0))
            )
        x0 = min(i.bbox[0] for i in on_page)
        y0 = min(i.bbox[1] for i in on_page)
        x1 = max(i.bbox[0] + i.bbox[2] for i in on_page)
        y1 = max(i.bbox[1] + i.bbox[3] for i in on_page)
        size = sum(i.bbox[2] for i in on_page) / len(on_page)
        for _ in range(3):
            bbox = (rng.uniform(x0, x1 - size), rng.uniform(y0, y1 - size), size, size)
            result.instances.append(
                Instance(file, page, rng.choice(symbols), bbox, score=rng.uniform(0.2, 0.8))
            )
        result.times[file] = result.times.get(file, 0.0) + rng.uniform(0.5, 2.0)
    return result


def print_summary(report) -> None:
    o = report.overall
    print(
        f"\nIoU ≥ {report.iou_threshold}, уверенность ≥ {report.score_threshold}\n"
        f"{'обозначение':<14}{'эталон':>7}{'TP':>5}{'FP':>5}{'FN':>5}"
        f"{'P':>8}{'R':>8}{'F1':>8}{'IoU':>8}{'AP':>8}"
    )
    for s in [*report.per_symbol, o]:
        print(
            f"{s.symbol:<14}{s.n_truth:>7}{s.tp:>5}{s.fp:>5}{s.fn:>5}"
            f"{fmt(s.precision):>8}{fmt(s.recall):>8}{fmt(s.f1):>8}{fmt(s.mean_iou):>8}"
            f"{fmt(s.ap):>8}"
        )
    print(f"macro F1 {fmt(report.macro['f1'])}, mAP {fmt(report.macro['map'])}")
    if report.timing:
        t = report.timing
        print(
            f"время: {t.mean_per_file_s:.3f} с на чертёж (медиана {t.median_per_file_s:.3f}, "
            f"макс. {t.max_per_file_s:.3f}), {t.mean_per_page_s:.3f} с на лист"
        )


if __name__ == "__main__":
    raise SystemExit(main())
