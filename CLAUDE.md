# SymSpot

Поиск обозначений на чертежах по эталонам с того же чертежа, с раскраской линий символов.
Полный план и архитектура: `docs/PLAN.md` — читать перед любой задачей.

## Роли
- Главная сессия (Opus) — оркестратор: контракты, алгоритмическое ядро, интеграция, ревью.
- Реализация периферии и тесты делегируются сабагентам на Sonnet (`.claude/agents/`):
  `sonnet-implementer`, `sonnet-tester`, `sonnet-viz`. Правила делегирования — `docs/PLAN.md`, раздел 0.

## Команды
- Установка: `uv sync`
- Тесты: `uv run pytest -q`
- Линт: `uv run ruff check . && uv run ruff format --check .`
- Синтетика: `uv run python -m symspot.synth --n 20 --size small --seed 0 --out data/synth`
- Запуск: `uv run symspot run <drawing> --queries <coco.json> --out out/`

## Соглашения
- Python 3.11, типы везде, dataclass/pydantic, без глобального состояния.
- `symspot/core/types.py` меняет только главная сессия.
- Все пороги — в `core/config.py`, никаких магических чисел в коде.
- Координаты: единицы чертежа, ось Y вниз (как в PDF/SVG после чтения).
- Каждый модуль покрыт тестами в `tests/` с тем же путём.
- Данные `data/real/` не коммитить.
- Коммиты маленькие, по одной задаче; сообщение: `phaseN/taskID: что сделано`.
