# FinAI-Qwen — Project Documentation

> **This is the primary reference document for the project.** Any new feature or
> structural change **must** be recorded here (and in
> [`docs/changelog.md`](docs/changelog.md)). If you read only one document, read
> this one.

---

## 1. What this project is

FinAI-Qwen is a self-contained, production-grade toolkit that lets the FinAI
platform **serve and fine-tune the Qwen2.5 model as a temporary stand-in** until
the in-house model (a separate project, *FinantalLM*) is ready. It is fully
independent of that project and shares no code with it.

It supports the complete lifecycle:

1. Download the model from HuggingFace.
2. Persist the full model to Google Drive.
3. Load the model directly from Drive.
4. Verify/inspect the model.
5. Run a streaming chat UI.
6. Fine-tune with QLoRA.
7. Resume an interrupted training run.
8. Merge the LoRA adapter into the base model.
9. Serve the merged model.
10. Evaluate (perplexity + qualitative) and benchmark (speed/memory).

It targets **Google Colab Pro+** with **Google Drive** as the persistence layer,
and is deployed via **GitHub** (clone on Colab, run).

### A note on the model name

`Qwen/Qwen2.5-8B` does not exist. The Qwen2.5 family is 0.5B/1.5B/3B/7B/14B/32B/72B;
`8B` exists only in Qwen3. The default is therefore `Qwen/Qwen2.5-7B-Instruct`,
changeable via `FINAI_MODEL_ID` or `--model-id` (e.g. `Qwen/Qwen3-8B`). Because
the model id is the only model-specific knob, swapping models is a one-line
change — see §10.

---

## 2. How to start from scratch

**On Colab (the intended path):**

1. Push this repository to your GitHub account.
2. Open `notebooks/FinAI_Qwen_Colab.ipynb` in Colab; select a GPU runtime.
3. Run the cells in order: GPU check → mount Drive → clone → install → set
   `FINAI_HOME` → download → verify → chat/train/merge/benchmark/evaluate.

**Locally (for development / tests):**

```bash
git clone <your-fork> && cd FinAI-Qwen
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # verify the pure-Python core
python scripts/verify_model.py --help
```

Without a GPU you can still run the test suite, the model verifier (on an
already-downloaded model) and all `--help` output; training/chat/merge need CUDA.

---

## 3. Folder-by-folder purpose

| Folder | Purpose |
|--------|---------|
| `config/` | **Single source of truth.** Path layout and all configuration dataclasses. Depends only on the standard library. No other module hard-codes a path or hyper-parameter. |
| `src/` | The reusable library. Contains all real logic, organised by concern. |
| `src/data/` | Dataset format detection/conversion (`formats.py`) and loading/splitting (`loader.py`). |
| `src/models/` | Download (`download.py`), load (`loader.py`), and inspect (`verify.py`) models. |
| `src/inference/` | The generation engine (`engine.py`) used by chat, benchmark and eval. |
| `src/training/` | QLoRA trainer (`trainer.py`) and LoRA merge (`merge.py`). |
| `src/evaluation/` | Perplexity + qualitative sampling (`evaluator.py`). |
| `src/utils/` | Cross-cutting helpers: environment detection, logging, system metrics, runtime bootstrap. |
| `scripts/` | Thin argparse CLIs that wire the library to the command line. |
| `data/samples/` | Real, small example datasets in every supported format. |
| `tests/` | Pytest suite covering the dependency-light core (no GPU). |
| `notebooks/` | The Colab notebook driving the full workflow. |
| `docs/` | Detailed topic documentation. |
| `logs/`, `outputs/` | Local placeholders (real runtime output goes to `FINAI_HOME`). |

---

## 4. File-by-file function

### `config/`
- **`paths.py`** — `ProjectPaths` (immutable layout), `get_paths()` (cached
  singleton), environment resolution (`FINAI_HOME` → Drive → ephemeral → repo),
  `configure_hf_cache()`, `sanitize_model_id()`.
- **`model_config.py`** — `ModelConfig`: model id, dtype, 4-bit/bnb settings,
  attention implementation (with `auto` resolution), download patterns; lazily
  builds `BitsAndBytesConfig` and maps dtypes.
- **`generation_config.py`** — `GenerationSettings`: sampling knobs and system
  prompt; `to_kwargs()` maps to `generate()` arguments; greedy when temp = 0.
- **`training_config.py`** — `LoRAConfig` and `TrainingConfig`: every training
  hyper-parameter, precision auto-resolution, effective-batch-size helper.
- **`__init__.py`** — re-exports the public configuration API.

### `src/utils/`
- **`bootstrap.py`** — `setup()`: put repo on `sys.path`, configure caches,
  create directories, log warnings. Called first by every entry point.
- **`env.py`** — Colab/Drive detection, Drive mounting, CUDA/bf16/flash-attn
  detection, `set_seed()`. Degrades gracefully without torch.
- **`logging_utils.py`** — `get_logger()` with de-duplicated console + rotating
  file handlers.
- **`system.py`** — VRAM/RAM stats, peak-memory reset, human-readable bytes.

### `src/data/`
- **`formats.py`** — `detect_format()` and converters that normalise every
  supported format into the unified `{"messages": [...]}` / `{"text": ...}` shape.
- **`loader.py`** — read files (`.jsonl`/`.json`, file or directory), normalise,
  render with the chat template, split into train/eval → `DatasetBundle`.

### `src/models/`
- **`download.py`** — `download_model()` (resumable, verified, skip-if-complete),
  `model_is_complete()`, `verify_download()`.
- **`loader.py`** — `load_model_and_tokenizer()` and friends: local-first
  resolution, quantisation, attention, k-bit training prep, padding sides.
- **`verify.py`** — `build_report()`: parameter count and dtype breakdown read
  straight from safetensors headers (no weights loaded), plus config/tokenizer/
  generation-config and integrity checks.

### `src/inference/`
- **`engine.py`** — `InferenceEngine`: `generate()` (blocking) and `stream()`
  (token-by-token via `TextIteratorStreamer` + background thread), chat-template
  input building, token counting.

### `src/training/`
- **`trainer.py`** — `run_training()`: load → attach LoRA → prepare dataset →
  build `SFTTrainer` → train (with resume) → save adapter. Version-tolerant
  config building (filters/renames kwargs to fit the installed TRL/Transformers).
- **`merge.py`** — `merge_adapter()`: load base in fp16/bf16, apply adapter,
  `merge_and_unload()`, save a standalone model (+ tokenizer + provenance).

### `src/evaluation/`
- **`evaluator.py`** — `compute_perplexity()`, `sample_generations()`,
  `evaluate()` → `EvalReport`.

### `scripts/`
- **`download_model.py`**, **`verify_model.py`**, **`train.py`**,
  **`merge_lora.py`**, **`benchmark.py`**, **`evaluate.py`** — each is a thin
  CLI: import shim → `setup()` → parse args → call the library → print results.

### Root
- **`chat.py`** — Gradio app (`build_demo()` + `main()`), streaming responses.
- **`requirements.txt`** / **`pyproject.toml`** — dependencies + tooling config.
- **`.gitignore`** / **`.env.example`** / **`LICENSE`** — repo hygiene.

---

## 5. How it works internally (the pipelines)

### Configuration & bootstrap
Every entry point begins with a two-line `sys.path` shim, then `setup()` from
`src/utils/bootstrap.py`. `setup()` resolves `ProjectPaths` (which decides where
everything lives), creates the base directories, redirects the HuggingFace cache
onto Drive, and emits warnings (e.g. Drive not mounted). From that point on, any
module can call `get_paths()` and receive the same cached layout.

### Download → verify
`download_model()` checks `model_is_complete()` first (instant, offline). If
incomplete, it calls `snapshot_download()` with `allow_patterns` (weights +
tokenizer + config only), which resumes partial transfers and skips matching
files, then verifies the shard set. `build_report()` inspects the result by
parsing safetensors headers directly — so it reports exact parameter counts and
dtypes without ever allocating the weights.

### Inference
`InferenceEngine` resolves the local model first (falling back to the Hub),
loads it (4-bit by default, left-padded for generation), and builds inputs with
`tokenizer.apply_chat_template(..., add_generation_prompt=True)`. `stream()`
runs `model.generate` on a background thread feeding a `TextIteratorStreamer`,
yielding text deltas the Gradio UI accumulates live.

### Training (QLoRA)
`run_training()` loads the base model 4-bit and `prepare_model_for_kbit_training`
(input grads on, KV cache off), attaches a LoRA adapter on the Qwen projection
layers, renders the dataset to a `text` column via the chat template, and builds
a TRL `SFTTrainer`. `SFTConfig` is assembled from `TrainingConfig` and then
**filtered against the installed library's accepted fields** (handling renames
like `evaluation_strategy`→`eval_strategy`), which is what keeps the trainer
working across "latest stable" version drift. Training uses gradient
accumulation/checkpointing, bf16 (auto), periodic eval, TensorBoard logging and
`save_total_limit` to bound checkpoints. `--resume` finds the latest checkpoint
via `get_last_checkpoint`. The final adapter, tokenizer and a config snapshot are
saved to `adapters/<run>`.

### Merge → serve
`merge_adapter()` loads the base in fp16/bf16 (never on the 4-bit base), applies
the adapter with PEFT, calls `merge_and_unload()`, and writes a complete,
standalone model to `merged/<run>-merged`. It can merge on CPU for small GPUs.
The merged directory is served exactly like the base model via
`chat.py --model-path ...`.

### Evaluate / benchmark
`evaluate()` computes token-weighted perplexity over a dataset and generates
sample answers. `benchmark.py` measures total latency, TTFT, throughput
(tokens/sec) and peak VRAM/RAM, averaged over warm runs with CUDA
synchronisation for accuracy.

---

## 6. Architecture decisions & rationale

| Decision | Why |
|----------|-----|
| **Config-only paths/params** | Reproducibility and portability: the same code runs on Colab+Drive or locally with zero edits; tests can redirect everything with one env var. |
| **`config/` depends only on stdlib** | It can be imported anywhere (tests, download script) without pulling in torch/transformers; keeps import time and failure surface low. |
| **Single `src/` package** (instead of separate top-level `training/`, `inference/`, `utils/`) | One import namespace, no `sys.path` gymnastics between sibling packages, clearer dependency direction. The requested concerns still exist as sub-packages. |
| **Library + thin CLIs** | Logic is testable and reusable; scripts only parse args and print. The notebook and other tools call the same functions. |
| **Local-first model resolution** | Avoids re-hitting the Hub once the model is on Drive; makes Colab restarts cheap. |
| **Safetensors header parsing for verification** | Exact parameter/dtype reporting with no GPU and no weight allocation. |
| **Version-tolerant trainer construction** | TRL/Transformers rename constructor args between releases; filtering against the installed signatures avoids brittle pins while honouring "latest stable". |
| **Auto-detected bf16 / flash-attn** | Best performance on capable GPUs (A100/L4) with safe fallbacks (fp16/SDPA) on others (T4), without user intervention. |
| **Merge in fp16/bf16, optional CPU** | Merging into a quantised base corrupts weights; CPU merge keeps it possible on small GPUs. |
| **Unified message schema** | Downstream code never branches on the original dataset format. |

More in [docs/project_architecture.md](docs/project_architecture.md).

---

## 7. Best practices used

- Strict typing (`from __future__ import annotations`, precise signatures, no
  gratuitous `Any`), dataclasses for all configuration and reports.
- Lazy imports of heavy/optional dependencies inside functions, so importing a
  module never forces torch/transformers/gradio to load.
- Single-responsibility modules; no duplicated logic (shared helpers in
  `src/utils`).
- Graceful degradation (no GPU, no Drive, no flash-attn, missing psutil).
- Idempotent, resumable operations (download, training).
- Reproducibility: global seeding, config snapshots saved with each run.
- Professional logging to console + rotating files.
- Tests for the deterministic core; runnable on CPU/CI.

---

## 8. How to add a new feature

1. **Put the logic in `src/`** under the right sub-package (create a new one if
   the concern is genuinely new; add an `__init__.py` that re-exports its API).
2. **Read configuration from `config/`** — add fields to the relevant dataclass
   (with an env-var default) rather than hard-coding. Never introduce a literal
   path; derive it from `ProjectPaths` (add a method/property if needed).
3. **Expose it via a thin CLI** in `scripts/` if it should be runnable, following
   the existing shim → `setup()` → argparse → call → print pattern.
4. **Add a test** in `tests/` for any pure-Python behaviour.
5. **Add a notebook cell** if it belongs in the Colab workflow.
6. **Document it**: update this file (§3/§4/§5 as appropriate), the relevant
   `docs/` page, and add an entry to [docs/changelog.md](docs/changelog.md).

---

## 9. How to maintain the project

- **Dependencies:** `requirements.txt` uses version floors. To freeze a known-good
  environment, pin exact versions (`==`) after a successful Colab run.
- **Version drift:** if a new TRL/Transformers release renames a constructor
  argument, extend `_CONFIG_ALIASES` in `src/training/trainer.py` (new→old name).
- **Logs/checkpoints:** retention is bounded by `save_total_limit`; old `runs/`
  and `logs/` on Drive can be pruned manually.
- **Tests:** run `pytest` before every commit; they need no GPU.
- **Secrets:** never commit tokens. Use `--token` or the `HF_TOKEN` env var.

---

## 10. How to update or replace the model

Because the model id is the only model-specific input, switching is trivial:

- **Different Qwen size or a fine-tune of Qwen:** set `FINAI_MODEL_ID` (or pass
  `--model-id`) and re-run download → verify. LoRA target modules already cover
  the Qwen2/2.5/3 architectures.

  ```bash
  export FINAI_MODEL_ID=Qwen/Qwen3-8B
  python scripts/download_model.py && python scripts/verify_model.py
  ```

- **A non-Qwen model** (the eventual FinantalLM, or Llama/Mistral/etc.): set the
  id the same way. If its architecture uses different projection layer names,
  update `QWEN_LORA_TARGET_MODULES` in `config/training_config.py` (or pass a
  custom `target_modules`). Base (non-instruct) models without a chat template
  require plain-text data or a template you set on the tokenizer — see
  [docs/dataset_formats.md](docs/dataset_formats.md).

- **Swapping the interim model out entirely** (when FinantalLM is ready): point
  `FINAI_MODEL_ID` at the new checkpoint, or serve a merged directory via
  `chat.py --model-path`. No other code changes are required for serving.

---

## 11. Important notes for any developer

- The local Windows build directory is irrelevant; the code never depends on a
  drive letter. On Colab everything lives under `FINAI_HOME` (Drive).
- 4-bit training/inference **requires a CUDA GPU**; the loaders fall back to full
  precision on CPU but that is for development only.
- Always run from the **repository root** (the scripts' `sys.path` shim assumes
  it; the notebook `%cd`s into it).
- Do not store large weights in Git — `.gitignore` blocks `models/`,
  `merged/`, `*.safetensors`, etc. Weights live on Drive/Hub, not in the repo.
- Keep this document and `docs/changelog.md` up to date with every change.

---

## 12. Document maintenance rule

Every change to the project — new file, new flag, new pipeline stage, renamed
config field, dependency bump with behavioural impact — must be reflected here
and appended to [docs/changelog.md](docs/changelog.md). Treat the documentation
as part of the definition of done.
