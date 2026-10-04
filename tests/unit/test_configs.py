"""Проверка всех конфигов команды: если кто-то добавил битый YAML, CI упадёт."""

import pytest

from peft_lab.config import (
    load_adapter_config,
    load_base_model_config,
    load_forgetting_config,
    load_training_config,
)
from peft_lab.registry import (
    list_adapters,
    list_base_models,
    list_forgetting_methods,
    list_training_methods,
)


@pytest.mark.parametrize("name", list_base_models())
def test_base_model_config_loads(name):
    cfg = load_base_model_config(name)
    assert "/" in cfg.hf_id, "hf_id должен иметь вид org/model"
    assert cfg.lora_target_modules


@pytest.mark.parametrize("name", list_training_methods())
def test_training_config_loads(name):
    cfg = load_training_config(name)
    assert cfg.method in {"lora", "qlora", "dora"}
    assert cfg.r > 0


@pytest.mark.parametrize("name", list_forgetting_methods())
def test_forgetting_config_loads(name):
    assert load_forgetting_config(name).method in {"none", "replay", "ewc"}


@pytest.mark.parametrize("name", list_adapters())
def test_adapter_config_resolves_references(name):
    cfg = load_adapter_config(name)
    assert cfg.base_model.name in list_base_models()
    assert cfg.output_dir.startswith("outputs/")


def test_templates_not_in_registry():
    assert "_template" not in list_base_models()
    assert "_template" not in list_adapters()


def test_unknown_config_raises():
    with pytest.raises(FileNotFoundError):
        load_base_model_config("no-such-model")
