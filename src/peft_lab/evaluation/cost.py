"""Контроль затрат на обучение: GPU-память, время, FLOPs (этап 3)."""


class CostTracker:
    """Контекстный менеджер: with CostTracker() as cost: train(...); cost.report()."""

    def __enter__(self):
        raise NotImplementedError("Этап 3: torch.cuda.reset_peak_memory_stats, time.perf_counter")

    def __exit__(self, *exc):
        return False

    def report(self) -> dict[str, float]:
        raise NotImplementedError("Этап 3")
