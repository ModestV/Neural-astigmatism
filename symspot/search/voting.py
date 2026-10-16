"""3.3 Голосование за преобразование. Ядро (Opus)."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Hypothesis, Query
from symspot.normalize.index import PrimIndex
from symspot.search.pairs import PairTable


def find_candidates(
    query: Query, table: PairTable, index: PrimIndex, cfg: Config
) -> list[Hypothesis]:
    """Гипотезы T (нормализованный эталон -> чертеж) для одного эталона.

    - Совпавший ключ пары -> подобие по двум якорям (для концентричных пар — по центру,
      отношению размеров и направлению); при ``allow_mirror`` то же для отраженного эталона.
    - Голос в аккумулятор: центр (шаг ``shift_bin`` диагонали эталона), угол
      (``angle_bin_deg``), log-масштаб (``scale_bin_log``), зеркало. Масштаб вне
      ``[scale_min, scale_max]`` отбрасывается.
    - Ячейка принимается, если голоса пришли от >= ``min_distinct_pairs`` разных пар эталона;
      вес голоса нормируется на локальную плотность кусков.
    - Для эталонов с текстом — дополнительные гипотезы от совпадающих строк текста.
    - NMS по центру, не больше ``max_hypotheses_per_query``, по убыванию голосов.
    """
    raise NotImplementedError
