"""Оценка качества поиска: загрузка разметки, сопоставление с ответами алгоритма, метрики.

    dataset.py   разметка и ответы в едином виде, загрузчики форматов
    metrics.py   IoU, precision, recall, F1, PR-кривая, AP, время
    run.py       прогон алгоритма по чертежам из разметки
    report.py    сводка в Markdown, PR-кривая в CSV и SVG

Командная строка — scripts/evaluate.py.
"""

from .dataset import Annotations, Instance, load, save
from .metrics import Report, evaluate, iou

__all__ = ["Annotations", "Instance", "Report", "evaluate", "iou", "load", "save"]
