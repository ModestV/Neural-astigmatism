"""Чтение SVG (svgelements). Задача T2."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import Drawing


def read_svg(path: str | Path) -> Drawing:
    """Прочитать SVG в ``Drawing(kind="vector")``.

    - Координаты в пользовательских единицах корневого ``viewBox`` (если он есть), иначе в px;
      все transform применены, ``<use>`` развернут, ось Y вниз.
    - Каждый геометрический элемент разбивается на связные подпути -> ``ParentPath``
      (line/quad/cubic; дуги SVG переводятся в cubic), ``group`` = порядковый номер элемента,
      ``source_ref`` = ``"svg:#<id>"`` если у элемента есть id, иначе ``"svg:<номер>"``.
    - ``<text>`` -> ``TextPrim`` (bbox по позиции и font-size, если точный недоступен)
      + ``ParentPath`` с одним item ``text``.
    - ``width``/``height`` = размер viewBox, ``units_per_px`` = 1.0.
    """
    raise NotImplementedError
