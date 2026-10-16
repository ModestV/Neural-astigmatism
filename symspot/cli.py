"""CLI (typer). Фаза 4."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="SymSpot: поиск обозначений на чертежах")


@app.command()
def run() -> None:
    """Заглушка: реализуется в фазе 4."""
    raise NotImplementedError
