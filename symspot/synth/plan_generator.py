"""Генерация плана с размещенными символами. Задача T1."""

from __future__ import annotations

from typing import Literal


def generate_plan(seed: int, size: Literal["small", "large"]) -> object:
    """Детерминированный по seed план (стены, двери, размеры, штриховки, текст) +
    экземпляры символов + легенда (по одному эталону каждого класса)."""
    raise NotImplementedError
