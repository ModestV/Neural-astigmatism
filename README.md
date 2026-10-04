# Neural-astigmatism

Анализ и практическое применение технологий fine-tune: локальное параметрически-эффективное
дообучение (PEFT) LLM с гарантированной приватностью данных.

> **Статус:** каркас проекта. Есть структура папок, документация и шаблоны конфигов, кода пока нет.

## Цель

- Изучить и применить **LoRA / QLoRA / DoRA**, включая их математику.
- **Переносимые адаптеры**: одна базовая модель переключается между ролями
  (программист, юрист, репетитор) подгрузкой разных LoRA-адаптеров; адаптеры можно комбинировать.
- **Контроль catastrophic forgetting**: дообучение не должно разрушать базовые знания модели
  (replay-буфер, EWC), эффект измеряется и показывается на графиках.
- **Учёт затрат**: GPU-память, время, FLOPs.
- **Локальное приложение**: данные пользователя не уходят в облако.

## Результат

- MVP приложения для локального дообучения LLM
- Модуль измерения качества дообучения и затрат на обучение
- Научно-техническая статья, открытый репозиторий с воспроизводимыми экспериментами и весами адаптеров

## Структура

```
configs/           YAML-описания экспериментов (сейчас только шаблоны _template.yaml)
  base_models/     базовые LLM: каждый участник может добавить свою
  adapters/        адаптеры: базовая модель + датасет + метод + защита от забывания
  training/        методы обучения: lora, qlora, dora
  forgetting/      защита от забывания: none, replay, ewc
  eval/            бенчмарки и метрики
src/peft_lab/      основной пакет: models, data, training, forgetting, adapters, evaluation, tracking
scripts/           точки входа: train, evaluate, export_ollama
app/               локальный UI на Gradio
data/              датасеты (не в git), формат описан в data/README.md
experiments/       личные папки участников: experiments/<github-username>/
notebooks/         общие исследовательские ноутбуки
tests/             unit/ и integration/
docs/              архитектура, теория PEFT
paper/             статья и графики
outputs/           обученные адаптеры и чекпоинты (создаётся при обучении, не в git)
```

В каждой папке есть `README.md`: что в ней будет и на каком этапе.

## Этапы

- [ ] 1. Обзор PEFT-методов (LoRA, QLoRA, DoRA) и их математики — `docs/theory/`
- [ ] 2. Выбор базовой модели и подбор датасета — `configs/base_models/`, `data/`
- [ ] 3. Пайплайн обучения с контролем catastrophic forgetting — `src/peft_lab/`
- [ ] 4. Переносимость и комбинирование адаптеров — `src/peft_lab/adapters/`
- [ ] 5. UI-приложение для локального использования — `app/`
- [ ] 6. Статья — `paper/`

## Начало работы

Рекомендуется Python 3.11 или 3.12.

```bash
git clone https://github.com/ModestV/Neural-astigmatism.git
cd Neural-astigmatism
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

make install          # основные зависимости (или make install-gpu на Linux + NVIDIA)
make install-dev      # линтер и тесты
cp .env.example .env  # и впишите токены
```

Без `make` (например, на Windows): команды смотрите в `Makefile`.

## Командная работа

- Порядок работы (ветки, PR, как добавить свою модель или адаптер) — [CONTRIBUTING.md](CONTRIBUTING.md).
- Правила для ИИ-ассистентов (Claude Code, Cursor, Codex, ChatGPT) — [AGENTS.md](AGENTS.md).
  Claude Code, Cursor и Codex подхватывают их автоматически; в ChatGPT вставьте `AGENTS.md` в начало диалога.

## Стек

PyTorch, HuggingFace Transformers, PEFT, TRL, bitsandbytes, Unsloth, Ollama, MLflow,
Weights & Biases, Gradio, Git.

## Источники

- Hu E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022
- Dettmers T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023
- Kirkpatrick J. et al. *Overcoming catastrophic forgetting in neural networks.* PNAS 2017
- [HuggingFace LLM Course](https://hf.co/learn), документация [PEFT](https://huggingface.co/docs/peft) и [Unsloth](https://docs.unsloth.ai)

## Лицензия

MIT
