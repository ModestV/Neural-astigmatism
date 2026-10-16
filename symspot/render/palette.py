"""Стабильная палитра по legend_id. Задача T4."""

from __future__ import annotations


def color_for(legend_id: str) -> tuple[int, int, int]:
    """Различимый RGB (0..255), детерминированный по ``legend_id`` (не зависит от порядка
    эталонов и от PYTHONHASHSEED)."""
    raise NotImplementedError
