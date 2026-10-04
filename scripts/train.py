"""Обучение адаптера по конфигу.

Пример: python scripts/train.py --adapter programmer
"""

import argparse

from peft_lab.config import load_adapter_config
from peft_lab.registry import list_adapters
from peft_lab.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение LoRA-адаптера")
    parser.add_argument(
        "--adapter", required=True, choices=list_adapters(), help="configs/adapters/<имя>.yaml"
    )
    args = parser.parse_args()

    cfg = load_adapter_config(args.adapter)
    print(f"Адаптер {cfg.name}: {cfg.base_model.hf_id}")
    print(f"Метод: {cfg.training.method}, защита от забывания: {cfg.forgetting.method}")
    path = train(cfg)
    print(f"Сохранён в {path}")


if __name__ == "__main__":
    main()
