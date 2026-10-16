"""Поиск точных копий эталона (без поворота и масштаба) по каноническому хешу. Задача T6."""

from __future__ import annotations

from symspot.core.config import Config
from symspot.core.types import Hypothesis, Query
from symspot.normalize.index import PrimIndex


def find_exact(query: Query, index: PrimIndex, cfg: Config) -> list[Hypothesis]:
    """Неповернутые немасштабированные копии эталона.

    - Канонический вид набора примитивов: координаты в единицах чертежа относительно
      якоря, округленные до ``exact_tol * query.diag``, отсортированные; отрезки без
      учета направления, дуги по (центр, r, t0, t1), текст по (центр, строка).
    - Якорь = самый редкий на чертеже тип/размер примитива эталона; для каждого такого
      примитива чертежа берется окно (рамка эталона, приложенная к якорю), и сравнивается
      хеш окна. Окно может содержать лишние куски (стена через символ): сравнение по
      вхождению всех примитивов эталона, а не по равенству множеств.
    - Каждая копия -> Hypothesis(T = Similarity(query.diag, 0, cx, cy), source="exact",
      votes = число примитивов эталона). Сам эталон тоже находится.
    """
    raise NotImplementedError
