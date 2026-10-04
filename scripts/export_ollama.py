"""Экспорт адаптера для Ollama: merge с базовой моделью -> GGUF -> Modelfile (этап 5).

Пример: python scripts/export_ollama.py --adapter programmer
"""

import argparse

from peft_lab.registry import list_adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт адаптера в Ollama")
    parser.add_argument("--adapter", required=True, choices=list_adapters())
    parser.parse_args()
    raise NotImplementedError("Этап 5")


if __name__ == "__main__":
    main()
