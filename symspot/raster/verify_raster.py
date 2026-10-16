"""Проверка гипотез на растре (переиспользует ядро фазы 3). Фаза 5, Opus."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Query, QuerySpec
from symspot.raster.preprocess import RasterPrep


def extract_raster_query(spec: QuerySpec, prep: RasterPrep, cfg: Config) -> Query:
    """Эталон из скелета в рамке: сэмплы = пиксели скелета (прореженные), касательные из
    ``prep.direction``, нормализация как в векторной ветке."""
    raise NotImplementedError
