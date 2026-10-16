"""3.8 Расширение запроса уверенными экземплярами. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Instance, Query
from symspot.normalize.index import PrimIndex


def expand_queries(
    instances: list[Instance], queries: dict[str, list[Query]], index: PrimIndex, cfg: Config
) -> list[Query]:
    """Экземпляры со score >= ``expand_score`` и остатком ICP > ``expand_residual`` становятся
    дополнительными эталонами своего класса (``variant`` > 0): эталон строится из кусков
    экземпляра, переведенных в нормализованную систему через T^-1. Не больше одного нового
    варианта на класс (самый уверенный).
    """
    raise NotImplementedError
