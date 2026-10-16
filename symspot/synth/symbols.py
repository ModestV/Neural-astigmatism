"""Библиотека синтетических символов. Задача T1."""

from __future__ import annotations

from dataclasses import dataclass

from symspot.core.types import Primitive


@dataclass(frozen=True)
class SymbolDef:
    """Символ в локальной системе: центр (0, 0), размер порядка 1 (диагональ рамки ~1).

    ``prims`` — Segment/Arc/TextPrim (id локальные, parent_id = номер штриха символа);
    ``group`` — имя группы похожих символов или None.
    """

    legend_id: str
    label: str
    prims: tuple[Primitive, ...]
    group: str | None = None


def symbol_library() -> dict[str, SymbolDef]:
    """>= 12 символов, включая группы похожих (см. PLAN.md T1)."""
    raise NotImplementedError
