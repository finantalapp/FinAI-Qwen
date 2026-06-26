# Folder Structure

```
FinAI-Qwen/
├── config/                      # Single source of truth (stdlib only)
│   ├── __init__.py              #   public configuration API
│   ├── paths.py                 #   ProjectPaths, get_paths(), FINAI_HOME resolution
│   ├── model_config.py          #   ModelConfig (model id, dtype, 4-bit, attention)
│   ├── generation_config.py     #   GenerationSettings (sampling + system prompt)
│   └── training_config.py       #   TrainingConfig + LoRAConfig
│
├── src/                         # Core library
│   ├── __init__.py
│   ├── data/
│   │   ├── formats.py           #   format detection + converters → unified schema
│   │   └── loader.py            #   read/normalise/render/split → DatasetBundle
│   ├── models/
│   │   ├── download.py          #   resumable, verified snapshot download
│   │   ├── loader.py            #   load model + tokenizer (local-first, quantised)
│   │   └── verify.py            #   safetensors-header model report
│   ├── inference/
│   │   └── engine.py            #   InferenceEngine (generate + stream)
│   ├── training/
│   │   ├── trainer.py           #   QLoRA SFT pipeline (+ resume)
│   │   └── merge.py             #   LoRA merge → standalone model
│   ├── evaluation/
│   │   └── evaluator.py         #   perplexity + qualitative sampling
│   └── utils/
│       ├── bootstrap.py         #   setup(): path/cache/dirs init
│       ├── env.py               #   colab/drive/cuda/bf16/flash-attn detection
│       ├── logging_utils.py     #   get_logger()
│       └── system.py            #   VRAM/RAM metrics
│
├── scripts/                     # CLI entry points (thin wrappers over src/)
│   ├── download_model.py
│   ├── verify_model.py
│   ├── train.py
│   ├── merge_lora.py
│   ├── benchmark.py
│   └── evaluate.py
│
├── chat.py                      # Gradio chat application
│
├── data/
│   └── samples/                 # Real example datasets, one per format
│       ├── alpaca_sample.jsonl
│       ├── sharegpt_sample.jsonl
│       ├── chatml_sample.jsonl
│       └── openai_sample.jsonl
│
├── tests/                       # Pytest suite (no GPU required)
│   ├── conftest.py
│   ├── test_formats.py
│   ├── test_config.py
│   └── test_data_loader.py
│
├── notebooks/
│   └── FinAI_Qwen_Colab.ipynb   # Full end-to-end Colab workflow
│
├── docs/                        # This documentation set
├── logs/    .gitkeep            # Local placeholder (runtime logs go to FINAI_HOME)
├── outputs/ .gitkeep            # Local placeholder
│
├── PROJECT_DOCUMENTATION.md     # Primary reference
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
└── LICENSE
```

## Runtime directories (created under `FINAI_HOME`, not in the repo)

These are produced at run time and are **git-ignored**; on Colab they live on
Google Drive:

| Directory      | Created by            | Contents                          |
|----------------|-----------------------|-----------------------------------|
| `models/`      | `download_model.py`   | Base model weights + tokenizer    |
| `datasets/`    | you                   | Your training/eval data           |
| `checkpoints/` | `train.py`            | Per-run training checkpoints      |
| `adapters/`    | `train.py`            | Final LoRA adapters               |
| `merged/`      | `merge_lora.py`       | Merged standalone models          |
| `runs/`        | `train.py`            | TensorBoard event files           |
| `logs/`        | all                   | Rotating application logs          |
| `cache/`       | all                   | HuggingFace hub/datasets cache    |

## Conventions

- Every package has an `__init__.py` that re-exports its public API, so callers
  import from the package (`from src.models import build_report`) rather than
  reaching into modules.
- Module names are nouns describing their concern; functions are verbs.
- Each entry point starts with a two-line `sys.path` shim and then calls
  `setup()`.
