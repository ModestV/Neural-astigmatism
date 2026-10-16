"""Метрики по экземплярам и классам. Задача T3."""

from __future__ import annotations

from dataclasses import dataclass, field

from symspot.core.types import GroundTruth, Instance


@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class Metrics:
    """Итог сравнения. ``confusion[gt_class][pred_class]`` — число сопоставленных пар
    (включая диагональ); ``"__none__"`` — несопоставленные (строка: ложные предсказания,
    столбец: пропуски). ``similar_group_error`` — доля GT из групп похожих символов,
    сопоставленных с предсказанием другого класса той же группы."""

    overall: ClassMetrics
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]
    similar_group_error: float = 0.0
    n_similar: int = 0
    matches: list[tuple[int, int, float]] = field(default_factory=list)  # (pred_i, gt_j, iou)

    def to_dict(self) -> dict:
        raise NotImplementedError


def evaluate(
    pred: list[Instance],
    gt: list[GroundTruth],
    iou_thr: float = 0.5,
    similar_groups: list[set[str]] | None = None,
    ignore_queries: bool = True,
) -> Metrics:
    """Венгерское сопоставление по IoU рамок (без учета класса), пары с IoU < ``iou_thr``
    отбрасываются. Совпадение класса -> TP; несовпадение -> FP предсказанного класса и FN
    класса GT (и запись в confusion). При ``ignore_queries`` экземпляры и GT с ``is_query``
    исключаются."""
    raise NotImplementedError
