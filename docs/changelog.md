# Changelog

All notable changes to FinAI-Qwen are recorded here. This project adheres to the
spirit of [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

> **Rule:** every change to the project must be added here and reflected in
> [PROJECT_DOCUMENTATION.md](../PROJECT_DOCUMENTATION.md).

## [0.1.0] — 2026-06-26

### Added
- **Configuration package** (`config/`): `ProjectPaths` with environment-aware
  `FINAI_HOME` resolution (Colab/Drive/local), and `ModelConfig`,
  `GenerationSettings`, `TrainingConfig`/`LoRAConfig` dataclasses with
  environment-variable defaults. No hard-coded paths anywhere in the codebase.
- **Utilities** (`src/utils/`): runtime bootstrap, environment detection
  (Colab, Drive, CUDA, bf16, flash-attention), logging, and VRAM/RAM metrics.
- **Data** (`src/data/`): auto-detecting converters for OpenAI, ShareGPT,
  Alpaca, ChatML, prompt/completion and plain-text formats into a unified schema,
  plus a loader that renders via the chat template and splits train/eval.
- **Models** (`src/models/`): resumable, verified, skip-if-present download;
  local-first model/tokenizer loading with 4-bit quantisation and attention
  auto-selection; a safetensors-header-based model report (parameter count,
  dtypes, sizes, integrity) requiring no torch.
- **Inference** (`src/inference/`): `InferenceEngine` with blocking and streaming
  generation built on the chat template.
- **Training** (`src/training/`): QLoRA pipeline (TRL `SFTTrainer`) with gradient
  accumulation/checkpointing, bf16 auto-detection, evaluation, TensorBoard,
  bounded checkpoint retention and resume; version-tolerant config construction;
  LoRA merge to a standalone model (GPU or CPU).
- **Evaluation** (`src/evaluation/`): token-weighted perplexity and qualitative
  sample generation.
- **CLI scripts** (`scripts/`): `download_model`, `verify_model`, `train`,
  `merge_lora`, `benchmark`, `evaluate`.
- **Chat app** (`chat.py`): streaming Gradio UI with history, system prompt and
  full sampling controls.
- **Sample datasets** (`data/samples/`): real finance examples in all four core
  formats.
- **Tests** (`tests/`): pure-Python coverage of formats, configuration, path
  resolution and record rendering (no GPU required).
- **Colab notebook** (`notebooks/FinAI_Qwen_Colab.ipynb`): full end-to-end
  workflow.
- **Documentation**: `README.md`, `PROJECT_DOCUMENTATION.md` and the `docs/`
  set (architecture, folder structure, training/inference pipelines, dataset
  formats, deployment, roadmap, changelog).
- **Tooling**: `requirements.txt`, `pyproject.toml` (ruff/black/mypy/pytest
  config), `.gitignore`, `.env.example`, `LICENSE`.

### Notes
- Default model is `Qwen/Qwen2.5-7B-Instruct` because `Qwen/Qwen2.5-8B` does not
  exist on the HuggingFace Hub (8B exists only in Qwen3). The model is selectable
  via `FINAI_MODEL_ID` / `--model-id`.
- The originally suggested top-level `training/`, `inference/`, `utils/`
  directories were consolidated under a single `src/` package for cleaner imports
  and an acyclic dependency graph; the same concerns remain as sub-packages.
