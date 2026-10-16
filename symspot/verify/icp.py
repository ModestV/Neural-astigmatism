"""3.4а ICP-уточнение гипотезы. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Query, Similarity
from symspot.normalize.index import PrimIndex


def icp_refine(
    query: Query, T: Similarity, index: PrimIndex, cfg: Config
) -> tuple[Similarity, float]:
    """Уточнить T: сэмплы эталона -> ближайшие сэмплы чертежа (согласованные касательные,
    < ``icp_angle_deg``), Umeyama с сохранением ``T.mirror``, до ``icp_iters`` итераций.

    Возвращает (T, residual), residual — средний остаток по сопоставленным сэмплам в долях
    диагонали эталона (inf, если сопоставлений меньше 3).
    """
    raise NotImplementedError
