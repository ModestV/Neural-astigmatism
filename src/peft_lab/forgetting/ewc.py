"""Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017), этап 3.

Штраф: L = L_task + (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2,
где F — диагональ матрицы Фишера на общих данных, theta* — веса до дообучения.
"""


def compute_fisher(model, dataloader, num_samples: int):
    """Диагональ матрицы Фишера по обучаемым параметрам."""
    raise NotImplementedError("Этап 3")


def ewc_penalty(model, fisher, reference_params, ewc_lambda: float):
    """EWC-штраф, который добавляется к loss на каждом шаге."""
    raise NotImplementedError("Этап 3")
