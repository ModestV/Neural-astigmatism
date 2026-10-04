"""Оценка адаптера: общие бенчмарки (забывание), доменные метрики, сравнение с базовой моделью.

Пример: python scripts/evaluate.py --adapter programmer
"""

import argparse

from peft_lab.config import load_adapter_config, read_yaml
from peft_lab.registry import list_adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка адаптера")
    parser.add_argument("--adapter", required=True, choices=list_adapters())
    parser.add_argument("--eval-config", default="configs/eval/default.yaml")
    args = parser.parse_args()

    cfg = load_adapter_config(args.adapter)
    eval_cfg = read_yaml(args.eval_config)
    raise NotImplementedError(f"Этап 3: оценка {cfg.name} по {eval_cfg['general_benchmarks']}")


if __name__ == "__main__":
    main()
