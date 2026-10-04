"""Загрузка базовой модели и токенизатора (этап 2–3)."""

from peft_lab.config import BaseModelConfig, TrainingConfig


def load_tokenizer(model_cfg: BaseModelConfig):
    """Токенизатор с chat template базовой модели."""
    raise NotImplementedError("Этап 3: AutoTokenizer.from_pretrained(model_cfg.hf_id)")


def load_base_model(model_cfg: BaseModelConfig, training_cfg: TrainingConfig):
    """Базовая модель; при training_cfg.load_in_4bit — квантизация NF4 через bitsandbytes."""
    raise NotImplementedError("Этап 3: AutoModelForCausalLM + BitsAndBytesConfig")
