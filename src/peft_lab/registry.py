"""Реестр моделей и адаптеров команды: всё, что лежит в configs/, без правки кода.

Файлы с именем, начинающимся на "_" (шаблоны), в реестр не попадают.
"""

from __future__ import annotations

from peft_lab.config import CONFIGS_DIR


def _list(kind: str) -> list[str]:
    return sorted(p.stem for p in (CONFIGS_DIR / kind).glob("*.yaml") if not p.stem.startswith("_"))


def list_base_models() -> list[str]:
    return _list("base_models")


def list_adapters() -> list[str]:
    return _list("adapters")


def list_training_methods() -> list[str]:
    return _list("training")


def list_forgetting_methods() -> list[str]:
    return _list("forgetting")
