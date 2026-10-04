"""Загрузка YAML-конфигов: базовые модели, адаптеры, режимы обучения и защиты от забывания."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class BaseModelConfig:
    name: str
    hf_id: str
    owner: str
    max_seq_length: int = 2048
    gated: bool = False
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    name: str
    method: str  # lora | qlora | dora
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = False
    use_dora: bool = False
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4


@dataclass
class ForgettingConfig:
    name: str
    method: str  # none | replay | ewc
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterConfig:
    name: str
    owner: str
    domain: str
    dataset: str
    base_model: BaseModelConfig
    training: TrainingConfig
    forgetting: ForgettingConfig
    output_dir: str


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался YAML-словарь")
    return data


def _load_named(kind: str, name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / kind / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Конфиг {kind}/{name}.yaml не найден")
    data = read_yaml(path)
    if data.get("name") != name:
        raise ValueError(f"{path}: поле name должно совпадать с именем файла ({name!r})")
    return data


def load_base_model_config(name: str) -> BaseModelConfig:
    return BaseModelConfig(**_load_named("base_models", name))


def load_training_config(name: str) -> TrainingConfig:
    return TrainingConfig(**_load_named("training", name))


def load_forgetting_config(name: str) -> ForgettingConfig:
    return ForgettingConfig(**_load_named("forgetting", name))


def load_adapter_config(name: str) -> AdapterConfig:
    """Собирает полный конфиг адаптера, подставляя базовую модель и режимы по именам."""
    raw = _load_named("adapters", name)
    return AdapterConfig(
        name=raw["name"],
        owner=raw["owner"],
        domain=raw["domain"],
        dataset=raw["dataset"],
        base_model=load_base_model_config(raw["base_model"]),
        training=load_training_config(raw.get("training", "lora")),
        forgetting=load_forgetting_config(raw.get("forgetting", "none")),
        output_dir=raw.get("output_dir", f"outputs/adapters/{raw['base_model']}/{raw['name']}"),
    )
