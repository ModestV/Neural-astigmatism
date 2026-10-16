"""Векторный и растровый конвейер. Фаза 4, Opus."""

from __future__ import annotations

from pathlib import Path

from symspot.core.config import Config
from symspot.core.types import Drawing, Result


def run(
    drawing_path: str | Path,
    queries_path: str | Path,
    cfg: Config,
    page: int = 0,
) -> tuple[Result, Drawing]:
    """Прочитать чертеж и эталоны, найти экземпляры. Возвращает (Result, нормализованный
    Drawing для рендеринга)."""
    raise NotImplementedError
