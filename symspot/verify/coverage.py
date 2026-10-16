"""3.4б Проверка гипотезы покрытием. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Hypothesis, Instance, Query
from symspot.normalize.index import PrimIndex


def verify_hypothesis(
    query: Query, hyp: Hypothesis, index: PrimIndex, cfg: Config
) -> Instance | None:
    """ICP + покрытие. Возвращает Instance (без piece_ids) или None при отказе.

    - Покрытие = взвешенная доля сэмплов эталона, у которых есть сэмпл чертежа ближе
      ``cover_tol`` * диагональ экземпляра + половина толщины линии, с согласованной
      касательной.
    - Отказ, если покрытие какого-то примитива эталона < ``min_prim_coverage``.
    - Текст эталона должен найтись на трансформированном месте с той же нормализованной
      строкой (при ``text_match``), иначе отказ.
    - score = weighted_coverage - ``extra_ink_lambda`` * extra_ink_ratio; отказ при
      score < ``accept_score``.
    """
    raise NotImplementedError
