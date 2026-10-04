# AGENTS.md

Instructions for AI coding assistants (Claude Code, Cursor, Codex, ChatGPT and others) working in this repository.
This is the single source of truth for AI assistants; `CLAUDE.md` imports this file. Keep it up to date when
the structure or conventions change.

## Project

**Neural-astigmatism** is a team research project: "Analysis and practical application of fine-tuning technologies".
We build an application for **local** parameter-efficient fine-tuning (PEFT) of LLMs, so a user can train a
personal assistant on their own data without sending anything to the cloud.

Goals:
1. Study and apply LoRA, QLoRA and DoRA, including their math.
2. **Portable adapters**: one base model switches between roles (programmer, lawyer, tutor) by loading different
   LoRA adapters; adapters can be combined.
3. **Catastrophic forgetting control**: fine-tuning must not destroy the base model's general knowledge
   (replay buffer, EWC), and the effect must be measured and plotted.
4. **Cost tracking**: GPU memory, training time, FLOPs.
5. A local UI (Gradio) and a scientific article with reproducible experiments.

Acceptance criteria: code in Git with unit and integration tests, documentation (README, architecture,
run instructions), a public MVP demo, and an article with measurable metrics.

## Current status

The repository is a **skeleton**: folder structure, documentation and config templates only.
**There is no Python code yet.** Each folder has a `README.md` describing what will live there. Read it before
adding code to that folder, and update it when the folder's contents change.

Stages (see `README.md`):
1. PEFT methods review and math → `docs/theory/`
2. Base model and dataset selection → `configs/base_models/`, `data/`, `src/peft_lab/data/`
3. Training pipeline with forgetting control → `src/peft_lab/{models,training,forgetting,evaluation,tracking}/`
4. Adapter portability and composition → `src/peft_lab/adapters/`
5. Local UI app → `app/`, `scripts/export_ollama.py`
6. Article → `paper/`

## Team model

Several people work in this repo, each possibly with **their own base model and their own adapters**.
Therefore:
- Everything experiment-specific is described in **YAML under `configs/`**, never hardcoded.
  Adding a base model or an adapter must require only a new YAML file, no code changes.
- Every config has an `owner` field (GitHub username). Do not modify configs owned by someone else unless asked.
- Personal drafts and notebooks go to `experiments/<github-username>/`; reusable code moves to `src/peft_lab/`.
- All changes reach `main` through pull requests.

## Repository map

```
configs/          YAML: base_models/, adapters/, training/, forgetting/, eval/  (files starting with "_" are templates)
src/peft_lab/     main package (models, data, training, forgetting, adapters, evaluation, tracking)
scripts/          CLI entry points: train.py, evaluate.py, export_ollama.py
app/              Gradio UI
data/             datasets — NOT in git (raw/, processed/, replay/); format in data/README.md
experiments/      personal folders: experiments/<github-username>/
notebooks/        shared research notebooks
tests/            unit/ (CPU, fast, run in CI) and integration/
docs/             architecture.md, theory/
paper/            article and figures/
outputs/          trained adapters and checkpoints — created at runtime, NOT in git
```

## Config contract

- File name equals the `name` field: `configs/adapters/programmer.yaml` → `name: programmer`.
- An adapter config references other configs **by name**: `base_model`, `training`, `forgetting`.
- Default adapter output path: `outputs/adapters/<base_model>/<adapter_name>/`.
- See the `_template.yaml` in each folder for the fields.

## Dataset format

`data/processed/<adapter>.jsonl`, one chat per line:
`{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`

## Tech stack

Python 3.11 (3.10–3.12 supported). PyTorch, HuggingFace Transformers, PEFT, TRL, Datasets, Accelerate,
bitsandbytes and Unsloth (GPU only), MLflow / Weights & Biases, Gradio, Ollama.
Dependencies are managed with **pip**:
- `requirements.txt` — core, must install on any OS without a GPU;
- `requirements-gpu.txt` — Linux + NVIDIA only (bitsandbytes, unsloth);
- `requirements-dev.txt` — ruff, pytest; must stay lightweight (no torch) because CI uses it.
When adding a dependency, put it in the right file with a minimum version (`>=`).

## Commands

```bash
make install        # core deps + package in editable mode
make install-gpu    # + bitsandbytes, unsloth
make install-dev    # ruff, pytest
make lint           # ruff check + ruff format --check
make format         # auto-fix
make test           # pytest -m "not gpu and not slow"
```

Run `make lint` and `make test` before proposing a commit.

## Coding conventions

- Code, identifiers, comments, docstrings and commit messages: **English**.
  User-facing documentation (`README.md`, `docs/`, folder READMEs): **Russian**.
- Ruff, line length 100, target Python 3.10. Type hints on public functions.
- Use `src/` layout: import as `from peft_lab... import ...` (the Python package is named `peft_lab`).
- Import heavy libraries (`torch`, `transformers`, `bitsandbytes`, `unsloth`) lazily or only in modules that need
  them, so that config code and unit tests run without them.
- Code must work on CPU (slowly) and must not crash at import time when there is no GPU.
  Detect the device at runtime; QLoRA / Unsloth paths are optional.
- Use dataclasses for configs; validate on load with clear error messages.
- Prefer small, focused modules and functions. No premature abstractions.

## Testing

- Unit tests in `tests/unit/`: fast, CPU-only, never download models or datasets from the internet.
  Use tiny fixtures or mocks.
- Mark tests that need a GPU with `@pytest.mark.gpu` and long ones with `@pytest.mark.slow`.
- Integration tests may use a tiny model (e.g. `sshleifer/tiny-gpt2` or a similar small checkpoint).
- Every new module comes with tests.

## Hard rules

- **Privacy is the core requirement.** The app and the pipeline must never send user data to external APIs or
  cloud LLMs. Network access is allowed only to download public models/datasets and, if enabled by the user,
  experiment tracking.
- Never commit: `.env`, datasets (`data/`), weights and checkpoints (`outputs/`, `*.safetensors`, `*.bin`,
  `*.pt`, `*.gguf`), `mlruns/`, `wandb/`. Trained adapters are shared via HuggingFace Hub.
- Never hardcode tokens or keys; read them from environment variables (see `.env.example`).
- Experiments must be reproducible: fixed seeds, all hyperparameters in configs, metrics logged to the tracker.
- Do not rewrite or delete other people's work (configs, experiments) without being asked.

## Domain glossary

- **PEFT** — parameter-efficient fine-tuning: train a small number of extra parameters, freeze the base model.
- **LoRA** — `W = W₀ + (α/r)·BA`, where `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, `r ≪ min(d, k)`; only A and B are trained.
- **QLoRA** — LoRA on top of a base model quantized to 4-bit NF4, with double quantization and paged optimizers.
- **DoRA** — weight-decomposed LoRA: weights split into magnitude and direction; LoRA updates the direction.
- **Adapter** — a saved set of LoRA weights for one domain/role, loadable on top of the same base model.
- **Catastrophic forgetting** — loss of general knowledge after fine-tuning on a narrow domain.
- **Replay** — mixing general-domain examples into the fine-tuning data.
- **EWC** — Elastic Weight Consolidation: penalty `(λ/2)·Σᵢ Fᵢ(θᵢ − θ*ᵢ)²`, F = diagonal Fisher information.

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017
- HuggingFace PEFT docs: https://huggingface.co/docs/peft ; Unsloth docs: https://docs.unsloth.ai
