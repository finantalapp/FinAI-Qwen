# Project Architecture

## Layered design

FinAI-Qwen is organised as a small set of layers with a strict, one-directional
dependency flow:

```
            scripts/  +  chat.py            (entry points: argparse / Gradio)
                     │  depend on
                     ▼
   src/inference  src/training  src/evaluation  src/models  src/data
                     │  depend on
                     ▼
                  src/utils                  (env, logging, system, bootstrap)
                     │  depend on
                     ▼
                  config/                    (paths + dataclasses, stdlib only)
```

- **Entry points** (`scripts/*`, `chat.py`) never contain business logic. They
  parse arguments, call `setup()`, invoke the library and print results.
- **`src/`** holds all reusable logic, split by concern.
- **`src/utils/`** provides cross-cutting helpers used by every other package.
- **`config/`** is the foundation. It depends only on the Python standard
  library, so it imports cleanly in any context (including CPU-only tests and the
  download script that has no need for torch).

Dependencies point **downward only**. `config/` knows nothing about `src/`;
`src/utils/` knows nothing about `src/training/`; entry points know about
everything. This keeps the graph acyclic and the layers independently testable.

> One intentional, contained exception: a couple of `config/` methods (e.g.
> `ModelConfig.resolve_attn_implementation`, `TrainingConfig.resolve_precision`)
> perform a **lazy, function-local** import of `src.utils.env` for hardware
> detection. Because it happens at call time (not import time), `config/` still
> imports with zero heavy dependencies.

## Runtime data flow

```
HuggingFace Hub ──download──▶ Drive: models/<name> ──load──▶ InferenceEngine ──▶ chat / benchmark / eval
                                          │
 dataset files ──normalise/render──▶ DatasetBundle ──▶ SFTTrainer ──▶ Drive: adapters/<run>
                                                                            │
                                                          merge ──▶ Drive: merged/<run>-merged ──▶ serve
```

## Key design properties

- **Configuration is centralised and environment-aware.** `ProjectPaths`
  resolves a single artefact root (`FINAI_HOME`) and derives every subdirectory
  from it. Switching between Colab+Drive and local development requires no code
  change.
- **Local-first model resolution.** `resolve_source()` prefers the downloaded
  copy on Drive and only falls back to the Hub, making restarts cheap and
  offline-friendly.
- **Graceful degradation.** Missing GPU, Drive, flash-attention or psutil are all
  handled with safe fallbacks rather than hard failures.
- **Version tolerance.** The trainer filters and remaps constructor arguments to
  match the installed TRL/Transformers, decoupling the code from a single pinned
  release.
- **Separation of weights from code.** Model weights live on Drive/Hub and are
  git-ignored; the repository stays small and clone-fast.

## Why a single `src/` package

The original brief suggested top-level `training/`, `inference/`, `utils/`
directories. Consolidating them under one `src/` package is a deliberate
improvement:

- a single import namespace (`src.*`) avoids `sys.path` juggling between sibling
  top-level packages;
- the dependency direction is explicit and easy to keep acyclic;
- the same conceptual separation is preserved as sub-packages
  (`src/training`, `src/inference`, ...).

See [folder_structure.md](folder_structure.md) for the concrete layout.
