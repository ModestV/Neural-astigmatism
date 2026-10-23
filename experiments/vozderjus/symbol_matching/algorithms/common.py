"""Общая геометрия и устранение повторных обнаружений."""

from __future__ import annotations

from models import Box, Detection


def iou(a: Box, b: Box) -> float:
    """Отношение площади пересечения рамок к площади объединения."""
    intersection = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x)) * max(
        0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    )
    return intersection / (a.w * a.h + b.w * b.h - intersection)


def nms(detections: list[Detection], threshold: float, limit: int) -> list[Detection]:
    """Подавить перекрывающиеся рамки одного класса, включая разные образцы."""
    kept = []
    for detection in sorted(detections, key=lambda d: d.score, reverse=True):
        if not any(
            d.symbol == detection.symbol and iou(d.bbox, detection.bbox) > threshold for d in kept
        ):
            kept.append(detection)
        if len(kept) >= limit:
            break
    return kept
