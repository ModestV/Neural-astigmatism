"""Пайплайн обучения адаптера: LoRA / QLoRA / DoRA + защита от забывания (этап 3)."""

from peft_lab.config import AdapterConfig


def build_peft_config(cfg: AdapterConfig):
    """peft.LoraConfig из cfg.training и cfg.base_model.lora_target_modules."""
    raise NotImplementedError("Этап 3")


def train(cfg: AdapterConfig) -> str:
    """Обучает адаптер и сохраняет его в cfg.output_dir. Возвращает путь к адаптеру.

    План: load_base_model -> датасет (+ replay) -> trl.SFTTrainer (+ EWC-штраф)
    -> трекинг метрик и стоимости -> save_pretrained(cfg.output_dir).
    """
    raise NotImplementedError("Этап 3")
