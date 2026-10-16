"""3.5а Отличающие части похожих эталонов. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Query


def apply_discriminative_weights(queries: list[Query], cfg: Config) -> list[Query]:
    """Для каждой пары эталонов схожего размера (отношение диагоналей <
    ``similar_size_ratio``) совмещает их (ICP с перебором поворотов/отражения); если они
    похожи (chamfer < ``similar_chamfer``), сэмплам одного, не покрытым другим, вес умножается
    на ``discriminative_weight``. Тексты эталона, которых нет в похожем эталоне, тоже
    считаются отличающими. Возвращает новые Query (исходные не меняются).
    """
    raise NotImplementedError
