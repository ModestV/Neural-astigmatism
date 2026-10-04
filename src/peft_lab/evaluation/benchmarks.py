"""Оценка качества: общие бенчмарки (деградация знаний) и доменные метрики (этап 3)."""


def evaluate_general(model, tokenizer, benchmarks: list[str], max_samples: int) -> dict[str, float]:
    """Точность на общих бенчмарках до и после дообучения — мера catastrophic forgetting."""
    raise NotImplementedError("Этап 3")


def evaluate_domain(model, tokenizer, dataset, metrics: list[str]) -> dict[str, float]:
    raise NotImplementedError("Этап 3")
