# `peft_lab` — основной пакет

Код пайплайна дообучения. Пока пусто: модули появятся по этапам проекта.

| Модуль | Что будет | Этап |
|---|---|---|
| `config` / `registry` | загрузка YAML из `configs/`, список моделей и адаптеров команды | 2 |
| [`data/`](data/) | загрузка датасетов, chat template | 2 |
| [`models/`](models/) | загрузка базовой модели, 4-bit квантизация | 3 |
| [`training/`](training/) | обучение LoRA / QLoRA / DoRA | 3 |
| [`forgetting/`](forgetting/) | защита от catastrophic forgetting: replay, EWC | 3 |
| [`evaluation/`](evaluation/) | бенчмарки, доменные метрики, стоимость обучения | 3 |
| [`tracking/`](tracking/) | логирование экспериментов: MLflow / W&B | 3 |
| [`adapters/`](adapters/) | загрузка, переключение и комбинирование адаптеров | 4 |
