# FinAI-Qwen

Production-ready toolkit for **serving and fine-tuning Qwen2.5** as the interim
model inside the FinAI platform, while a bespoke model is developed from scratch.
Everything is designed to run on **Google Colab Pro+** with all artefacts
persisted to **Google Drive**.

> **Model note.** `Qwen/Qwen2.5-8B` **does not exist** on the HuggingFace Hub.
> The Qwen2.5 family ships in 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B sizes; the
> `8B` size only exists in **Qwen3**. This project therefore defaults to the
> closest real checkpoint, **`Qwen/Qwen2.5-7B-Instruct`**, and lets you switch
> to any other model with a single setting (`FINAI_MODEL_ID` or `--model-id`),
> e.g. `Qwen/Qwen3-8B`.

---

## Features

- **One config, no hard-coded paths.** All paths and hyper-parameters live in
  `config/` and adapt automatically to Colab + Drive or a local machine.
- **Drive-first storage.** Model, tokenizer, checkpoints, logs, datasets,
  merged models — everything is written to Drive so nothing is lost on restart.
- **Resumable model download** with integrity verification and skip-if-present.
- **Streaming Gradio chat** with system prompt and full sampling controls.
- **QLoRA fine-tuning** (4-bit, LoRA) with gradient accumulation/checkpointing,
  bf16 auto-detection, flash-attention auto-detection, evaluation, TensorBoard,
  bounded checkpoint retention and **resume**.
- **LoRA merge** into a standalone full-precision model.
- **Benchmark** (tokens/sec, TTFT, latency, VRAM, RAM) and **evaluation**
  (perplexity + qualitative samples).
- **Multi-format datasets**: JSONL, ShareGPT, ChatML, Alpaca, OpenAI messages,
  prompt/completion — auto-detected and normalised.
- **Tested** pure-Python core and a step-by-step **Colab notebook**.

## Project structure

```
FinAI-Qwen/
├── config/            # Single source of truth: paths + model/generation/training configs
├── src/               # Core library
│   ├── data/          #   dataset format conversion + loading
│   ├── models/        #   download, load, verify
│   ├── inference/     #   generation engine (streaming)
│   ├── training/      #   QLoRA trainer + LoRA merge
│   ├── evaluation/    #   perplexity + qualitative eval
│   └── utils/         #   env detection, logging, system metrics, bootstrap
├── scripts/           # CLI entry points (download/verify/train/merge/benchmark/evaluate)
├── chat.py            # Gradio chat application
├── data/samples/      # Real example datasets in every supported format
├── tests/             # Pytest suite (no GPU required)
├── notebooks/         # FinAI_Qwen_Colab.ipynb — full end-to-end workflow
├── docs/              # Detailed documentation
├── PROJECT_DOCUMENTATION.md  # The single most important reference document
├── requirements.txt
└── pyproject.toml
```

See [docs/folder_structure.md](docs/folder_structure.md) for the rationale and a
file-by-file description.

## Installation

### Google Colab (recommended)

Open [`notebooks/FinAI_Qwen_Colab.ipynb`](notebooks/FinAI_Qwen_Colab.ipynb) in
Colab and run the cells top to bottom. It mounts Drive, clones your fork,
installs dependencies and walks through every stage.

### Local / server

```bash
git clone https://github.com/<your-username>/FinAI-Qwen.git
cd FinAI-Qwen
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is required for 4-bit loading, training and merging. The pure-Python
tests and the model verifier run on CPU.

## Configuration

Nothing is hard-coded. Configure via environment variables (see
[`.env.example`](.env.example)) or CLI flags. The most important one:

```bash
export FINAI_HOME=/content/drive/MyDrive/FinAI-Qwen   # where everything is stored
export FINAI_MODEL_ID=Qwen/Qwen2.5-7B-Instruct        # which model to use
```

On Colab with Drive mounted, `FINAI_HOME` defaults to
`/content/drive/MyDrive/FinAI-Qwen` automatically.

## Usage

All commands are run from the repository root.

### 1. Download the model (to Drive)

```bash
python scripts/download_model.py
# or a specific model / private repo:
python scripts/download_model.py --model-id Qwen/Qwen3-8B --token hf_xxx
```

Resumable and idempotent — re-running skips files already present.

### 2. Verify the model

```bash
python scripts/verify_model.py
```

Prints parameter count, dtype breakdown, size, per-file sizes, integrity,
tokenizer class and generation config.

### 3. Chat (Gradio)

```bash
python chat.py            # local
python chat.py --share    # public link (Colab)
```

Supports conversation history, streaming, system prompt, temperature, top-p,
top-k, repetition penalty and max tokens.

### 4. Train with QLoRA

```bash
# Smoke test on the bundled sample data:
python scripts/train.py --run-name finai-qlora \
    --dataset data/samples/openai_sample.jsonl --max-steps 50 --eval-ratio 0

# Full run on your own data:
python scripts/train.py --run-name finai-v1 --dataset /path/to/data.jsonl --epochs 3
```

### 5. Resume training

```bash
python scripts/train.py --run-name finai-v1 --resume
```

### 6. Merge the LoRA adapter

```bash
python scripts/merge_lora.py --adapter finai-v1
# small GPU? merge in host RAM:
python scripts/merge_lora.py --adapter finai-v1 --device cpu
```

### 7. Run / benchmark the merged model

```bash
python chat.py --model-path "$FINAI_HOME/merged/finai-v1-merged" --share
python scripts/benchmark.py --model-path "$FINAI_HOME/merged/finai-v1-merged"
```

### 8. Evaluate

```bash
python scripts/evaluate.py --model-path "$FINAI_HOME/merged/finai-v1-merged" \
    --dataset data/samples/openai_sample.jsonl
```

## Google Drive

Every persistent artefact is stored under `FINAI_HOME` (Drive on Colab):

| Subdirectory   | Contents                              |
|----------------|---------------------------------------|
| `models/`      | Downloaded base model + tokenizer     |
| `datasets/`    | Your training/eval datasets           |
| `checkpoints/` | Per-run training checkpoints          |
| `adapters/`    | Final LoRA adapters                   |
| `merged/`      | Merged standalone models              |
| `runs/`        | TensorBoard logs                      |
| `logs/`        | Application logs                      |
| `cache/`       | HuggingFace download/datasets cache   |

Colab's ephemeral `/content` is only used when Drive is unavailable (you'll be
warned).

## Dataset formats

JSONL / ShareGPT / ChatML / Alpaca / OpenAI messages / prompt-completion / plain
text are all auto-detected and normalised. See
[docs/dataset_formats.md](docs/dataset_formats.md) and the working examples in
[`data/samples/`](data/samples).

## Testing

```bash
pytest            # pure-Python tests, no GPU/torch needed
```

## Common errors & fixes

| Symptom | Fix |
|---------|-----|
| `Repository not found` / 401 on download | The model id is wrong (e.g. the non-existent `Qwen2.5-8B`) or gated. Use a real id and/or pass `--token`. |
| CUDA out of memory while training | Lower `--train-batch-size`, raise `--grad-accum`, reduce `--max-seq-len`. |
| CUDA OOM during merge | Use `python scripts/merge_lora.py --adapter <run> --device cpu`. |
| `bitsandbytes` / CUDA error | 4-bit needs a CUDA GPU; on CPU the loader falls back to full precision automatically. |
| flash-attn import/build errors | Leave it uninstalled — the project falls back to PyTorch SDPA. |
| Artefacts disappear after restart | Drive wasn't mounted; mount it or set `FINAI_HOME`. |
| `tokenizer has no chat template` | You used a base model with chat-format data. Use an `-Instruct` model or plain-text data. |

More detail in [docs/deployment.md](docs/deployment.md).

## Documentation

- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** — the primary reference.
- [docs/project_architecture.md](docs/project_architecture.md)
- [docs/folder_structure.md](docs/folder_structure.md)
- [docs/training_pipeline.md](docs/training_pipeline.md)
- [docs/inference_pipeline.md](docs/inference_pipeline.md)
- [docs/dataset_formats.md](docs/dataset_formats.md)
- [docs/deployment.md](docs/deployment.md)
- [docs/roadmap.md](docs/roadmap.md)
- [docs/changelog.md](docs/changelog.md)

## License

MIT for the project code (see [LICENSE](LICENSE)). The Qwen model weights are
governed by their own upstream license.
