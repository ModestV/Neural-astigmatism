"""3.5б Соревнование экземпляров разных классов. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Instance, Query
from symspot.normalize.index import PrimIndex


def compete(
    instances: list[Instance],
    queries: dict[str, list[Query]],
    index: PrimIndex,
    cfg: Config,
) -> list[Instance]:
    """Экземпляры разных классов с IoU рамок > ``compete_iou`` сравниваются: побеждает
    больший score с отрывом >= ``compete_margin`` (проигравший удаляется), иначе оба
    остаются со ``status="ambiguous"`` и взаимными ``rivals``. Score в соревновании учитывает
    отличающие части обоих эталонов (присутствие отличающих частей соперника на чертеже —
    свидетельство против).
    """
    raise NotImplementedError
