"""Объектная оценка: один экземпляр разметки соответствует одной детекции."""

from __future__ import annotations

from algorithms.common import iou
from models import Box, Detection, TargetSpec


def inside_region(box: Box, region: Box) -> bool:
    """Рамка относится к исключаемой области по положению её центра."""
    return (
        region.x <= box.x + box.w / 2 <= region.x + region.w
        and region.y <= box.y + box.h / 2 <= region.y + region.h
    )


def ignored_regions(target: TargetSpec, include_legend: bool) -> list[Box]:
    """Общие исключения и известные экземпляры легенды."""
    regions = list(target.ignore_regions)
    if not include_legend and target.ground_truth is not None:
        regions.extend(t.bbox for t in target.ground_truth if t.legend)
    return regions


def filter_detections(
    predictions: list[Detection],
    regions: list[Box],
) -> list[Detection]:
    return [d for d in predictions if not any(inside_region(d.bbox, r) for r in regions)]


def metrics(tp: int, fp: int, fn: int) -> dict:
    """Неопределённые precision/recall оставляем null, а не объявляем единицей."""
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def evaluate(
    predictions: list[Detection],
    target: TargetSpec,
    threshold: float,
    include_legend: bool,
) -> dict | None:
    """Жадное сопоставление по score, затем максимальному IoU в своём классе."""
    if target.ground_truth is None:
        return None
    regions = ignored_regions(target, include_legend)
    truth = [t for t in target.ground_truth if not any(inside_region(t.bbox, r) for r in regions)]
    predictions = filter_detections(predictions, regions)
    classes = sorted({t.symbol for t in truth} | {d.symbol for d in predictions})
    per_class = {}
    for symbol in classes:
        gt = [t for t in truth if t.symbol == symbol]
        pred = sorted(
            (d for d in predictions if d.symbol == symbol),
            key=lambda d: d.score,
            reverse=True,
        )
        unmatched = set(range(len(gt)))
        tp = 0
        for detection in pred:
            best = max(
                sorted(unmatched),
                key=lambda i: iou(detection.bbox, gt[i].bbox),
                default=None,
            )
            if best is not None and iou(detection.bbox, gt[best].bbox) >= threshold:
                tp += 1
                unmatched.remove(best)
        per_class[symbol] = metrics(tp, len(pred) - tp, len(gt) - tp)
    return {
        **metrics(*(sum(v[k] for v in per_class.values()) for k in ("tp", "fp", "fn"))),
        "per_class": per_class,
    }
