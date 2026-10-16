"""Метрики по экземплярам и классам. Задача T3."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from symspot.core.types import BBox, GroundTruth, Instance

NONE_CLASS = "__none__"
# Надбавка к IoU при сопоставлении пары одного класса (класс важнее мелкой разницы IoU).
SAME_CLASS_BONUS = 1.0


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

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class Metrics:
    """Итог сравнения. ``confusion[gt_class][pred_class]`` — число сопоставленных пар
    (включая диагональ); ``"__none__"`` — несопоставленные (строка: ложные предсказания,
    столбец: пропуски). ``similar_group_error`` — доля GT из групп похожих символов,
    сопоставленных с предсказанием другого класса той же группы.
    ``matches`` — (индекс в ``pred``, индекс в ``gt``, IoU) в исходных списках, переданных в
    ``evaluate`` (до отбрасывания ``is_query``)."""

    overall: ClassMetrics
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]
    similar_group_error: float = 0.0
    n_similar: int = 0
    matches: list[tuple[int, int, float]] = field(default_factory=list)  # (pred_i, gt_j, iou)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.to_dict(),
            "per_class": {c: m.to_dict() for c, m in sorted(self.per_class.items())},
            "confusion": {
                g: dict(sorted(row.items())) for g, row in sorted(self.confusion.items())
            },
            "similar_group_error": self.similar_group_error,
            "n_similar": self.n_similar,
        }


def _iou_matrix(a: list[BBox], b: list[BBox]) -> np.ndarray:
    """Матрица IoU (len(a), len(b)) для рамок (x0, y0, x1, y1)."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    ix0 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy0 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix1 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix1 - ix0, 0.0, None) * np.clip(iy1 - iy0, 0.0, None)
    area_a = np.clip(A[:, 2] - A[:, 0], 0.0, None) * np.clip(A[:, 3] - A[:, 1], 0.0, None)
    area_b = np.clip(B[:, 2] - B[:, 0], 0.0, None) * np.clip(B[:, 3] - B[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


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
    pred_idx = [i for i, p in enumerate(pred) if not (ignore_queries and p.is_query)]
    gt_idx = [j for j, g in enumerate(gt) if not (ignore_queries and g.is_query)]
    iou = _iou_matrix([pred[i].bbox for i in pred_idx], [gt[j].bbox for j in gt_idx])

    per_class: dict[str, ClassMetrics] = {}
    confusion: dict[str, dict[str, int]] = {}

    def cls(name: str) -> ClassMetrics:
        return per_class.setdefault(name, ClassMetrics())

    def conf(g: str, p: str) -> None:
        row = confusion.setdefault(g, {})
        row[p] = row.get(p, 0) + 1

    pairs: list[tuple[int, int]] = []  # индексы в отфильтрованных списках
    if iou.size:
        # при равных условиях предпочитаем пару того же класса
        same = np.array(
            [[pred[i].legend_id == gt[j].legend_id for j in gt_idx] for i in pred_idx], dtype=float
        )
        gain = iou + SAME_CLASS_BONUS * same * (iou >= iou_thr)
        rows, cols = linear_sum_assignment(-gain)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= iou_thr]

    matched_p = {r for r, _ in pairs}
    matched_g = {c for _, c in pairs}
    matches: list[tuple[int, int, float]] = []
    groups = [set(g) for g in (similar_groups or [])]
    n_similar = 0
    n_similar_err = 0

    for r, c in sorted(pairs):
        p, g = pred[pred_idx[r]], gt[gt_idx[c]]
        matches.append((pred_idx[r], gt_idx[c], float(iou[r, c])))
        conf(g.legend_id, p.legend_id)
        if p.legend_id == g.legend_id:
            cls(g.legend_id).tp += 1
        else:
            cls(p.legend_id).fp += 1
            cls(g.legend_id).fn += 1
        in_groups = [s for s in groups if g.legend_id in s]
        if in_groups:
            n_similar += 1
            if p.legend_id != g.legend_id and any(p.legend_id in s for s in in_groups):
                n_similar_err += 1

    for r, i in enumerate(pred_idx):
        if r not in matched_p:
            cls(pred[i].legend_id).fp += 1
            conf(NONE_CLASS, pred[i].legend_id)
    for c, j in enumerate(gt_idx):
        if c not in matched_g:
            cls(gt[j].legend_id).fn += 1
            conf(gt[j].legend_id, NONE_CLASS)

    overall = ClassMetrics(
        tp=sum(m.tp for m in per_class.values()),
        fp=sum(m.fp for m in per_class.values()),
        fn=sum(m.fn for m in per_class.values()),
    )
    return Metrics(
        overall=overall,
        per_class=per_class,
        confusion=confusion,
        similar_group_error=n_similar_err / n_similar if n_similar else 0.0,
        n_similar=n_similar,
        matches=matches,
    )
