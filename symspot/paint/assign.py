"""3.6 Назначение кусков экземпляру. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Instance, Query
from symspot.normalize.index import PrimIndex


def assign_pieces(instance: Instance, query: Query, index: PrimIndex, cfg: Config) -> Instance:
    """Заполнить ``piece_ids``/``parent_ids``: кусок принадлежит экземпляру, если
    >= ``assign_inside`` его сэмплов ближе допуска покрытия к трансформированному эталону
    и касательные согласованы. Тексты эталона, найденные на месте, тоже назначаются.
    """
    raise NotImplementedError
