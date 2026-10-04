# peft-lab

Анализ и практическое применение технологий fine-tune: локальное параметрически-эффективное
дообучение (PEFT) LLM с гарантированной приватностью данных.

- **LoRA / QLoRA / DoRA**: математика методов и сравнение на практике
- **Переносимые адаптеры**: одна базовая модель переключается между ролями
  (программист, юрист, репетитор) подгрузкой разных LoRA-адаптеров
- **Контроль catastrophic forgetting**: replay-буфер и EWC, метрики деградации базовых знаний
- **Учёт затрат**: GPU-память, время, FLOPs
- **Локальное приложение**: данные пользователя не уходят в облако

## Структура

```
configs/
  base_models/     базовые LLM: один YAML на модель (каждый участник может добавить свою)
  adapters/        адаптеры: базовая модель + датасет + метод + защита от забывания
  training/        lora.yaml, qlora.yaml, dora.yaml
  forgetting/      none.yaml, replay.yaml, ewc.yaml
  eval/            бенчмарки и метрики
src/peft_lab/      код пайплайна (config, registry, models, data, training,
                   forgetting, adapters, evaluation, tracking)
scripts/           train.py, evaluate.py, export_ollama.py
app/               Gradio UI
data/              датасеты (не в git), формат — data/README.md
experiments/       личные папки участников: experiments/<username>/
notebooks/         общие исследовательские ноутбуки
tests/             unit/ и integration/
docs/              архитектура, теория PEFT
paper/             статья и графики
outputs/           обученные адаптеры и чекпоинты (создаётся автоматически, не в git)
```

## Быстрый старт

Рекомендуется Python 3.11 или 3.12.

```bash
git clone https://github.com/semenenkoe964-lang/peft-lab.git
cd peft-lab
python -m venv .venv && source .venv/bin/activate

make install          # или make install-gpu на Linux + NVIDIA (QLoRA, Unsloth)
make install-dev      # линтер и тесты
cp .env.example .env  # и впишите токены

make test
make train ADAPTER=programmer
make app
```

Без `make` (например, на Windows): смотрите команды в `Makefile`.

## Этапы

- [ ] 1. Обзор PEFT-методов (LoRA, QLoRA, DoRA) и их математики — `docs/theory/`
- [ ] 2. Выбор базовой модели и подбор датасета — `configs/base_models/`, `data/`
- [ ] 3. Пайплайн обучения с контролем catastrophic forgetting — `training/`, `forgetting/`, `evaluation/`
- [ ] 4. Переносимость и комбинирование адаптеров — `adapters/`
- [ ] 5. UI-приложение для локального использования — `app/`, `scripts/export_ollama.py`
- [ ] 6. Статья — `paper/`

## Как участвовать

Читайте [CONTRIBUTING.md](CONTRIBUTING.md): ветки, PR, как подключить свою модель и обучить адаптер.

## Стек

PyTorch, HuggingFace Transformers, PEFT, TRL, bitsandbytes, Unsloth, Ollama, MLflow,
Weights & Biases, Gradio.

## Источники

- Hu E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022
- Dettmers T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023
- Kirkpatrick J. et al. *Overcoming catastrophic forgetting in neural networks.* PNAS 2017
- [HuggingFace LLM Course](https://hf.co/learn), документация [PEFT](https://huggingface.co/docs/peft) и [Unsloth](https://docs.unsloth.ai)

## Лицензия

MIT
