"""3.7 Глобальное разрешение конфликтов за куски. Ядро (Opus)."""

from __future__ import annotations

import dataclasses

from symspot.core.config import Config
from symspot.core.geometry_utils import bbox_diag, bbox_iou
from symspot.core.types import Instance, Query, TextPrim


def query_complexity(q: Query) -> float:
    """Сложность эталона: число сэмплов + тексты (по длине их рамок в шагах сэмплинга)."""
    extra = 0.0
    for p in q.prims:
        if isinstance(p, TextPrim):
            # текст считается как линия длиной в диагональ его рамки, в тех же сэмплах
            extra += bbox_diag(p.bbox) * max(1.0, len(q.samples)) / max(_length(q), 1e-9)
    return len(q.samples) + extra


def _length(q: Query) -> float:
    from symspot.verify.coverage import query_length

    return query_length(q)


def _overlap(a: Instance, b: Instance) -> float:
    """IoU множеств кусков (или рамок, если кусков нет — растр)."""
    if a.piece_ids and b.piece_ids:
        sa, sb = set(a.piece_ids), set(b.piece_ids)
        return len(sa & sb) / len(sa | sb)
    return bbox_iou(a.bbox, b.bbox)


def resolve(
    instances: list[Instance], queries: dict[str, list[Query]], cfg: Config
) -> list[Instance]:
    """Эталоны по убыванию сложности (числа сэмплов), внутри класса по убыванию score.
    Экземпляр забирает куски; следующий, у которого > ``resolve_taken_frac`` кусков уже заняты
    другим экземпляром, отклоняется. Дубли одного класса (IoU множеств кусков >
    ``dedup_iou``) схлопываются в один. Возвращает итоговые экземпляры с id подряд.
    """
    comp = {lid: max(query_complexity(q) for q in qs) for lid, qs in queries.items()}
    order = sorted(
        range(len(instances)),
        key=lambda k: (-comp.get(instances[k].legend_id, 0.0), -instances[k].score),
    )
    owner: dict[int, int] = {}  # piece_id -> индекс принятого экземпляра
    accepted: list[int] = []
    for k in order:
        inst = instances[k]
        # дубли своего класса
        if any(
            instances[a].legend_id == inst.legend_id
            and _overlap(instances[a], inst) > cfg.dedup_iou
            for a in accepted
        ):
            continue
        if inst.piece_ids:
            taken = 0
            for pid in inst.piece_ids:
                o = owner.get(pid)
                if o is None:
                    continue
                other = instances[o]
                # куски, занятые взаимным ambiguous-соперником, не считаются
                if (
                    inst.status == "ambiguous"
                    and other.status == "ambiguous"
                    and other.legend_id in inst.rivals
                ):
                    continue
                taken += 1
            if taken / len(inst.piece_ids) > cfg.resolve_taken_frac:
                continue
            for pid in inst.piece_ids:
                owner.setdefault(pid, k)
        accepted.append(k)
    accepted.sort(key=lambda k: (instances[k].bbox[1], instances[k].bbox[0]))
    return [dataclasses.replace(instances[k], id=n) for n, k in enumerate(accepted)]
