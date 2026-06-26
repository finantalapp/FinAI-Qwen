# Deployment & Operations

## Target environment

- **GitHub** for source (clone on Colab).
- **Google Colab Pro+** for compute (A100 / L4 recommended for the 7B model;
  T4 works with smaller settings and fp16).
- **Google Drive** for persistence of all artefacts.

## Colab workflow

Use [`notebooks/FinAI_Qwen_Colab.ipynb`](../notebooks/FinAI_Qwen_Colab.ipynb).
The stages are:

1. **GPU check** — `!nvidia-smi`.
2. **Mount Drive** — `drive.mount('/content/drive')`.
3. **Clone** your fork (edit `REPO_URL`); re-runs `git pull`.
4. **Install** — `pip install -r requirements.txt`.
5. **Set `FINAI_HOME`** to a Drive path (default already points there on Colab).
6. **Download → Verify → Chat → Train → Resume → Merge → Benchmark → Evaluate.**

## Storage layout on Drive

All under `FINAI_HOME` (default `/content/drive/MyDrive/FinAI-Qwen`):
`models/`, `datasets/`, `checkpoints/`, `adapters/`, `merged/`, `runs/`,
`logs/`, `cache/`. See [folder_structure.md](folder_structure.md).

Because the HuggingFace cache is redirected to `cache/` on Drive, a re-downloaded
model is reused across runtime restarts.

## GPU sizing guide

| Task | T4 (16 GB) | L4 (24 GB) | A100 (40/80 GB) |
|------|-----------|-----------|------------------|
| Chat / inference (4-bit 7B) | OK | OK | OK |
| QLoRA training (7B) | OK with small batch + seq-len, fp16 | Comfortable, bf16 | Best, large batch |
| Merge (fp16 7B) | Use `--device cpu` | OK on GPU | OK on GPU |
| flash-attention-2 | Not supported | Supported | Supported |

## Secrets

- For gated/private models, pass `--token hf_xxx` to `download_model.py` or set
  `HF_TOKEN` in the environment. **Never commit tokens.**

## Updating a deployment

```bash
cd FinAI-Qwen && git pull --ff-only
pip install -r requirements.txt        # if dependencies changed
```

Artefacts on Drive are unaffected by a code update.

## Common errors & fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Repository Not Found` / 401 on download | Wrong/gated model id (e.g. the non-existent `Qwen2.5-8B`) | Use a real id; pass `--token` for gated models |
| `CUDA out of memory` (training) | Batch/seq-len too large | Lower `--train-batch-size`, raise `--grad-accum`, reduce `--max-seq-len` |
| `CUDA out of memory` (merge) | fp16 base too big for the GPU | `merge_lora.py --device cpu` |
| `bitsandbytes` CUDA error | No CUDA / mismatched build | 4-bit needs a GPU; loaders fall back to full precision on CPU for dev |
| flash-attn build/import error | Unsupported GPU or missing build tools | Leave it uninstalled; SDPA is used automatically |
| Artefacts gone after restart | Drive not mounted | Mount Drive (step 2) or set `FINAI_HOME` |
| `tokenizer has no chat template` | Base model + chat data | Use `-Instruct` model or plain-text data |
| Slow first generation | Model loading / CUDA warmup | Expected; the benchmark warms up before measuring |

## Health checks

```bash
pytest                              # core logic
python scripts/verify_model.py      # downloaded model integrity & stats
python scripts/benchmark.py         # speed & memory sanity
```

## Serving notes

`chat.py` launches a Gradio server (`--host`, `--port`, `--share`). For a more
permanent serving setup behind the FinAI platform, the same `InferenceEngine` can
be wrapped in a FastAPI/uvicorn service or a high-throughput server (e.g. vLLM)
pointed at the **merged** model directory — see [roadmap.md](roadmap.md).
