# forgetting

Контроль catastrophic forgetting:
- **replay** — подмешивание общих данных (`data/replay/`) к доменным;
- **EWC** — Elastic Weight Consolidation: штраф `(λ/2)·Σ Fᵢ(θᵢ − θ*ᵢ)²` по матрице Фишера (Kirkpatrick et al., 2017).
