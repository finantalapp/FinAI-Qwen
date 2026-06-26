# Inference Pipeline

## Components

- **`src/inference/engine.py` — `InferenceEngine`**: the single generation path
  used by the chat UI, the benchmark and the evaluator.
- **`chat.py`**: the Gradio application built on top of the engine.

## Loading

`InferenceEngine.from_config()` loads the configured model (local copy on Drive
preferred, Hub fallback), 4-bit by default, with the tokenizer padded on the
**left** (correct for batched generation). `InferenceEngine.from_path()` loads an
explicit directory — typically a merged model — in full precision.

## Building inputs

Both generation methods call `build_inputs()`, which:

1. optionally prepends the system prompt (unless the conversation already starts
   with a `system` turn),
2. applies the tokenizer's **chat template** with `add_generation_prompt=True`,
3. moves the tensors to the model's device.

This guarantees the exact prompt format Qwen was trained on.

## Generation modes

### Blocking — `generate(messages, settings) -> str`
Runs `model.generate` under `torch.inference_mode()` and decodes only the newly
generated tokens (everything after the prompt).

### Streaming — `stream(messages, settings) -> Iterator[str]`
Creates a `TextIteratorStreamer`, runs `model.generate` on a **background
thread**, and yields text deltas as they are produced. The Gradio UI accumulates
these deltas to render a live, token-by-token response. The streamer thread is
always joined in a `finally` block.

## Sampling settings (`config/generation_config.py`)

| Setting | Default | Notes |
|---------|---------|-------|
| `max_new_tokens` | 1024 | Generation length cap |
| `temperature` | 0.7 | `0` ⇒ greedy (sampling disabled) |
| `top_p` | 0.8 | Nucleus sampling |
| `top_k` | 20 | Top-k sampling |
| `repetition_penalty` | 1.05 | Discourages loops |
| `system_prompt` | FinAI default | Prepended when absent |

`to_kwargs()` omits the sampling-only knobs when `do_sample` is `False`, avoiding
spurious "sampling parameter ignored" warnings.

## The chat application

```bash
python chat.py                                   # local
python chat.py --share                           # public link (Colab)
python chat.py --model-path "$FINAI_HOME/merged/finai-v1-merged" --share
python chat.py --no-quantize                     # full precision
```

The UI provides:

- conversation history (`type="messages"`),
- streaming responses,
- an editable **system prompt**,
- live sliders for max tokens, temperature, top-p, top-k and repetition penalty.

Each request rebuilds `GenerationSettings` from the slider values via
`with_overrides`, so changing a slider takes effect on the next message with no
reload.

## Serving the merged model

A merged model directory is a complete, standalone model (weights + tokenizer).
Serve it exactly like the base model with `--model-path`. No PEFT or bitsandbytes
is required at inference time for a merged model loaded in full precision.

## Programmatic use

```python
from src.inference.engine import InferenceEngine
from config.generation_config import GenerationSettings

engine = InferenceEngine.from_config()
print(engine.generate([{"role": "user", "content": "What is a bond?"}],
                      GenerationSettings(temperature=0.3)))
```
