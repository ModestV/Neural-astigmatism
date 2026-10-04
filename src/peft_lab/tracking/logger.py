"""Единый интерфейс логирования экспериментов: MLflow или Weights & Biases.

Бэкенд выбирается переменной TRACKING_BACKEND в .env (mlflow | wandb | none).
"""


def init_run(run_name: str, params: dict) -> None:
    raise NotImplementedError("Этап 3")


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    raise NotImplementedError("Этап 3")


def finish_run() -> None:
    raise NotImplementedError("Этап 3")
