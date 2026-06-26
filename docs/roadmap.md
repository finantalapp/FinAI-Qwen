# Roadmap

FinAI-Qwen is the **interim** model layer for the FinAI platform. Its job is to
provide a solid, production-quality serving + fine-tuning stack for Qwen2.5 until
the in-house model (FinantalLM) is ready to take over.

## Status

| Capability | State |
|------------|-------|
| Config-driven, Drive-backed storage | ✅ Done |
| Resumable model download + verification | ✅ Done |
| Streaming Gradio chat | ✅ Done |
| QLoRA training + resume + TensorBoard | ✅ Done |
| LoRA merge (GPU/CPU) | ✅ Done |
| Benchmark + evaluation | ✅ Done |
| Multi-format dataset ingestion | ✅ Done |
| Pure-Python test suite | ✅ Done |
| Colab notebook | ✅ Done |

## Planned / candidate work

### Serving
- FastAPI/uvicorn wrapper around `InferenceEngine` with an OpenAI-compatible
  `/v1/chat/completions` endpoint.
- High-throughput serving via **vLLM** or **TGI** pointed at the merged model.
- Request batching and a token-rate limiter.

### Training
- Train-on-completions-only (mask the prompt tokens from the loss).
- DPO/ORPO preference tuning stage after SFT.
- Multi-GPU / `accelerate` launch configuration.
- Dataset quality filtering and deduplication utilities.

### Evaluation
- Domain-specific financial benchmark suite.
- Automatic regression comparison between two model versions.

### Ops
- Pinned, lock-filed dependency set for fully reproducible Colab runs.
- CI workflow running `pytest` + `ruff` on pull requests.
- Optional Weights & Biases logging backend alongside TensorBoard.

### Model migration (the end goal)
- Swap the interim Qwen model for FinantalLM by changing `FINAI_MODEL_ID` (and,
  if the architecture differs, the LoRA target modules). See
  [PROJECT_DOCUMENTATION.md](../PROJECT_DOCUMENTATION.md) §10.

## Contributing a roadmap item

Pick an item, implement it following
[PROJECT_DOCUMENTATION.md](../PROJECT_DOCUMENTATION.md) §8 ("How to add a new
feature"), add tests and docs, and record it in
[changelog.md](changelog.md).
