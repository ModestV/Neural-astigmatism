# tests

- `unit/` — быстрые тесты без GPU и без скачивания моделей; запускаются в CI.
- `integration/` — тесты пайплайна целиком (загрузка модели, короткое обучение).

Маркеры pytest (заданы в `pyproject.toml`): `gpu` — нужен NVIDIA GPU, `slow` — долгий тест.
В CI запускается `pytest -m "not gpu and not slow"`.
