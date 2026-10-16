"""Чтение векторного PDF (PyMuPDF). Задача T2."""

from __future__ import annotations

from pathlib import Path

from symspot.core.types import Drawing


def read_pdf(path: str | Path, page: int = 0) -> Drawing:
    """Прочитать страницу PDF в ``Drawing(kind="vector")``.

    - ``page.get_drawings()``: каждый drawing разбивается на связные подпути (по разрывам
      между концом предыдущего item и началом следующего); каждый подпуть -> ``ParentPath``
      с items ``line``/``cubic``/``quad`` (``re`` и ``qu`` раскладываются на 4 line),
      ``group`` = индекс drawing, ``source_ref`` = ``"pdf:<индекс>"``, ``style`` из width/color,
      ``fill`` = есть заливка без обводки.
    - ``page.get_text("dict")``: каждый span -> ``TextPrim`` (+ ``ParentPath`` с одним item
      ``text`` и пустой полилинией); ``angle`` из ``line["dir"]``.
    - Координаты в pt, ось Y вниз, с учетом ``page.rotation`` (как на отображаемой странице,
      через ``page.rotation_matrix``). ``width``/``height`` = размер отображаемой страницы.
    - ``units_per_px`` = 1.0, ``source_path``, ``page`` заполнены, ``normalized=False``.
    - id родителей и TextPrim уникальны, начинаются с 0, идут подряд.
    """
    raise NotImplementedError
