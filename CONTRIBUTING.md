# Как работать в репозитории

## Ветки и PR

- `main` всегда рабочий; прямой push в `main` не делаем.
- Ветка на задачу: `<username>/<кратко-что-делаю>`, например `ivan/ewc-penalty`.
- Изменения попадают в `main` через Pull Request с хотя бы одним ревью и зелёным CI.
- Перед PR: `make lint` и `make test`.

## Подключить свою базовую модель

1. Скопируйте `configs/base_models/_template.yaml` в `configs/base_models/<name>.yaml`.
2. Заполните `name` (= имя файла), `hf_id`, `owner` (ваш GitHub-ник), `lora_target_modules`.

## Обучить свой адаптер

1. Положите датасет в `data/processed/<adapter>.jsonl` (формат — в `data/README.md`).
2. Скопируйте `configs/adapters/_template.yaml` в `configs/adapters/<adapter>.yaml` и укажите
   `base_model`, `training` (lora / qlora / dora), `forgetting` (none / replay / ewc).
3. Запустить обучение (команда появится на этапе 3).
4. Адаптер сохранится в `outputs/adapters/<base_model>/<adapter>/` (не в git).
   Чтобы поделиться им, опубликуйте его на HuggingFace Hub и укажите ссылку в PR.

## Чего не коммитить

`.env`, датасеты (`data/`), веса и чекпоинты (`outputs/`, `*.safetensors`, `*.gguf`),
логи `mlruns/` и `wandb/`. Всё это уже в `.gitignore`.

## Личные эксперименты

Черновики и ноутбуки: `experiments/<username>/`. Переиспользуемый код переносите в `src/peft_lab/`.

## ИИ-ассистенты

Правила для ИИ (Claude Code, Cursor, Codex, ChatGPT) — в [AGENTS.md](AGENTS.md); `CLAUDE.md` его подключает.
Cursor, Codex и Claude Code читают их автоматически. В ChatGPT (веб-версия) вставьте содержимое
`AGENTS.md` в начало диалога или в инструкции проекта.

Если меняете структуру репозитория или договорённости команды, обновите `AGENTS.md` в том же PR.
