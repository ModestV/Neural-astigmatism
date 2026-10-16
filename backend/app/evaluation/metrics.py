"""Метрики качества поиска: IoU, precision, recall, F1, PR-кривая и AP, время на чертёж.

Сопоставление ответов с эталоном — как в COCO:
  * ответы и эталоны группируются по (файл, лист, обозначение);
  * ответы перебираются по убыванию уверенности, каждый забирает ещё не занятый эталон
    с наибольшим IoU;
  * IoU ≥ порога — попадание (TP), иначе ложное срабатывание (FP);
  * эталоны, оставшиеся без пары, — пропуски (FN).

Экземпляры в таблице условных обозначений (legend) по умолчанию не оцениваются: из неё обычно
берут образец для поиска. Ответ, попавший в такой экземпляр, не считается ни TP, ни FP.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from .dataset import Annotations, BBoxTuple, Instance


def iou(a: BBoxTuple, b: BBoxTuple) -> float:
    """Intersection over Union двух рамок (x, y, w, h)."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class MatchedPrediction:
    prediction: Instance
    truth: Instance | None
    """Сопоставленный эталон; None — ложное срабатывание."""
    iou: float
    """IoU с сопоставленным эталоном или, для FP, с ближайшим эталоном того же типа."""


@dataclass
class MatchResult:
    predictions: list[MatchedPrediction] = field(default_factory=list)
    """Оцениваемые ответы (без попавших в игнорируемые эталоны), по убыванию уверенности."""
    missed: list[Instance] = field(default_factory=list)
    """Пропущенные эталоны (FN)."""
    n_truth: int = 0
    ignored: int = 0
    """Сколько ответов попало в игнорируемые эталоны (таблицу обозначений)."""


def match(
    truth: Annotations,
    predictions: Annotations,
    iou_threshold: float = 0.5,
    include_legend: bool = False,
) -> MatchResult:
    """Сопоставляет ответы алгоритма с эталоном (см. docstring модуля)."""
    groups: dict[tuple[str, int, str], tuple[list[Instance], list[Instance]]] = defaultdict(
        lambda: ([], [])
    )
    for inst in truth.instances:
        groups[inst.file, inst.page, inst.symbol][0].append(inst)
    for inst in predictions.instances:
        groups[inst.file, inst.page, inst.symbol][1].append(inst)

    result = MatchResult()
    for gts, preds in groups.values():
        scored = [g for g in gts if include_legend or not g.legend]
        ignored = [g for g in gts if not (include_legend or not g.legend)]
        result.n_truth += len(scored)
        free = list(scored)
        # sorted() устойчива: при равной уверенности сохраняется исходный порядок ответов
        for pred in sorted(preds, key=lambda p: -p.score):
            best, best_iou = _best(pred, free)
            if best is not None and best_iou >= iou_threshold:
                free.remove(best)
                result.predictions.append(MatchedPrediction(pred, best, best_iou))
                continue
            _, ignored_iou = _best(pred, ignored)
            if ignored and ignored_iou >= iou_threshold:
                result.ignored += 1
                continue
            _, nearest_iou = _best(pred, scored)
            result.predictions.append(MatchedPrediction(pred, None, nearest_iou))
        result.missed.extend(free)
    result.predictions.sort(key=lambda m: -m.prediction.score)
    return result


def _best(pred: Instance, candidates: Iterable[Instance]) -> tuple[Instance | None, float]:
    best, best_iou = None, 0.0
    for cand in candidates:
        value = iou(pred.bbox, cand.bbox)
        if value > best_iou:
            best, best_iou = cand, value
    return best, best_iou


# --- сводные метрики ----------------------------------------------------------


@dataclass
class Scores:
    """Метрики для одного обозначения или для всех вместе."""

    symbol: str
    n_truth: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    """None — алгоритм ничего не нашёл, точность не определена."""
    recall: float | None
    """None — в эталоне нет экземпляров."""
    f1: float | None
    mean_iou: float | None
    """Средний IoU по попаданиям."""
    ap: float | None
    """Average precision — площадь под PR-кривой."""


@dataclass
class PRPoint:
    threshold: float
    precision: float
    recall: float


@dataclass
class Timing:
    files: int
    pages: int
    total_s: float
    mean_per_file_s: float
    median_per_file_s: float
    max_per_file_s: float
    mean_per_page_s: float
    per_file_s: dict[str, float]


@dataclass
class Report:
    iou_threshold: float
    score_threshold: float
    include_legend: bool
    detector: str | None
    overall: Scores
    macro: dict[str, float | None]
    """Средние по обозначениям: precision, recall, f1, map."""
    per_symbol: list[Scores]
    pr_curve: list[PRPoint]
    timing: Timing | None
    ignored_predictions: int
    errors: list[dict]
    """Пропуски и ложные срабатывания — для разбора ошибок в отчёте."""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    truth: Annotations,
    predictions: Annotations,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    include_legend: bool = False,
) -> Report:
    """Считает все метрики.

    score_threshold отсекает неуверенные ответы для precision/recall/F1/IoU.
    PR-кривая и AP строятся по всем ответам — они и показывают, какой порог выбрать.
    """
    m = match(truth, predictions, iou_threshold, include_legend)
    symbols = sorted(
        {i.symbol for i in truth.instances} | {i.symbol for i in predictions.instances}
    )
    per_symbol = [
        _scores(
            symbol,
            [p for p in m.predictions if p.prediction.symbol == symbol],
            sum(1 for g in m.missed if g.symbol == symbol),
            score_threshold,
        )
        for symbol in symbols
    ]
    per_symbol = [s for s in per_symbol if s.n_truth or s.tp or s.fp]
    overall = _scores("all", m.predictions, len(m.missed), score_threshold)
    return Report(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        include_legend=include_legend,
        detector=predictions.detector,
        overall=overall,
        macro={
            "precision": _mean(s.precision for s in per_symbol),
            "recall": _mean(s.recall for s in per_symbol),
            "f1": _mean(s.f1 for s in per_symbol if s.n_truth),
            "map": _mean(s.ap for s in per_symbol if s.n_truth),
        },
        per_symbol=per_symbol,
        pr_curve=pr_curve(m.predictions, overall.n_truth),
        timing=timing(truth, predictions),
        ignored_predictions=m.ignored,
        errors=_errors(m, score_threshold),
    )


def _scores(
    symbol: str, matched: list[MatchedPrediction], missed: int, score_threshold: float
) -> Scores:
    # Ответы с уверенностью ниже порога отбрасываются, а их эталоны переходят в пропуски.
    kept = [p for p in matched if p.prediction.score >= score_threshold]
    dropped_tp = sum(1 for p in matched if p.truth is not None) - sum(
        1 for p in kept if p.truth is not None
    )
    tp = sum(1 for p in kept if p.truth is not None)
    fp = len(kept) - tp
    fn = missed + dropped_tp
    n_truth = tp + fn
    ious = [p.iou for p in kept if p.truth is not None]
    return Scores(
        symbol=symbol,
        n_truth=n_truth,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / n_truth if n_truth else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        mean_iou=statistics.fmean(ious) if ious else None,
        ap=average_precision(pr_curve(matched, n_truth)) if n_truth else None,
    )


def pr_curve(matched: list[MatchedPrediction], n_truth: int) -> list[PRPoint]:
    """Точки (порог, precision, recall) для каждого различного значения уверенности."""
    if n_truth == 0:
        return []
    ordered = sorted(matched, key=lambda p: -p.prediction.score)
    points: list[PRPoint] = []
    tp = fp = 0
    for n, item in enumerate(ordered):
        if item.truth is not None:
            tp += 1
        else:
            fp += 1
        last_of_score = (
            n + 1 == len(ordered) or ordered[n + 1].prediction.score != item.prediction.score
        )
        if last_of_score:
            points.append(PRPoint(item.prediction.score, tp / (tp + fp), tp / n_truth))
    return points


def average_precision(points: list[PRPoint]) -> float:
    """AP с интерполяцией по всем точкам (как в PASCAL VOC 2010+ и COCO).

    Площадь под «огибающей» PR-кривой: в каждой точке берётся максимальная точность
    при этой и большей полноте.
    """
    if not points:
        return 0.0
    ap, prev_recall = 0.0, 0.0
    for n, point in enumerate(points):
        envelope = max(p.precision for p in points[n:])
        ap += (point.recall - prev_recall) * envelope
        prev_recall = point.recall
    return ap


def timing(truth: Annotations, predictions: Annotations) -> Timing | None:
    if not predictions.times:
        return None
    per_file = dict(sorted(predictions.times.items()))
    values = list(per_file.values())
    pages = len({(f, p) for f, p in truth.pages() if f in per_file}) or len(per_file)
    return Timing(
        files=len(values),
        pages=pages,
        total_s=sum(values),
        mean_per_file_s=statistics.fmean(values),
        median_per_file_s=statistics.median(values),
        max_per_file_s=max(values),
        mean_per_page_s=sum(values) / pages,
        per_file_s=per_file,
    )


def _errors(m: MatchResult, score_threshold: float) -> list[dict]:
    errors = [
        {
            "type": "fp",
            "file": p.prediction.file,
            "page": p.prediction.page,
            "symbol": p.prediction.symbol,
            "bbox": list(p.prediction.bbox),
            "score": p.prediction.score,
            "iou": round(p.iou, 4),
        }
        for p in m.predictions
        if p.truth is None and p.prediction.score >= score_threshold
    ]
    missed = list(m.missed) + [
        p.truth
        for p in m.predictions
        if p.truth is not None and p.prediction.score < score_threshold
    ]
    errors += [
        {
            "type": "fn",
            "file": g.file,
            "page": g.page,
            "symbol": g.symbol,
            "bbox": list(g.bbox),
            "angle": g.angle,
        }
        for g in missed
    ]
    return errors


def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None
