"""Планаризация: разрез отрезков и дуг в точках пересечения. Задача T5."""

from __future__ import annotations

from collections.abc import Sequence

from symspot.core.types import Arc, Segment


def planarize(
    prims: Sequence[Segment | Arc],
    snap_tol: float,
    min_len: float,
    start_id: int = 0,
) -> list[Segment | Arc]:
    """Разрезать все примитивы во всех точках взаимного пересечения и касания концом.

    - Кандидаты пар через ``shapely.STRtree`` по рамкам; пересечения аналитически:
      отрезок-отрезок, отрезок-дуга, дуга-дуга (включая полные окружности).
    - Конец одного примитива, лежащий на другом ближе ``snap_tol`` (Т-стык), режет другой.
    - Каждый кусок: новый id подряд с ``start_id``, ``parent_id`` и ``style`` исходного,
      ``stroke_id`` = id исходного примитива. Куски короче ``min_len`` отбрасываются.
    - Кусок полной окружности -> Arc с t0 < t1 (t1 - t0 < 2π), если окружность разрезана
      хотя бы в двух точках; разрезанная в одной точке окружность остается полной.
    - Сложность ~ O(n log n + k). Порядок выхода детерминирован.
    """
    raise NotImplementedError
