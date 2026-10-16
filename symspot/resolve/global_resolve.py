"""3.7 Глобальное разрешение конфликтов за куски. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Instance, Query


def resolve(
    instances: list[Instance], queries: dict[str, list[Query]], cfg: Config
) -> list[Instance]:
    """Эталоны по убыванию сложности (числа сэмплов), внутри класса по убыванию score.
    Экземпляр забирает куски; следующий, у которого > ``resolve_taken_frac`` кусков уже заняты
    другим экземпляром, отклоняется. Дубли одного класса (IoU множеств кусков >
    ``dedup_iou``) схлопываются в один. Возвращает итоговые экземпляры с id подряд.
    """
    raise NotImplementedError
