"""QLoRA fine-tuning configuration.

The configuration is split into two dataclasses:

* :class:`LoRAConfig` - the PEFT/LoRA adapter hyper-parameters.
* :class:`TrainingConfig` - the optimisation loop, dataset and checkpoint
  settings, plus an embedded :class:`LoRAConfig`.

Both are plain dataclasses with environment-variable-aware defaults, so a run
can be configured either by editing this file, by exporting variables, or by
passing command-line flags (see ``scripts/train.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

# LoRA target modules for the Qwen2 / Qwen2.5 / Qwen3 architectures. These are
# the attention and MLP projection layers; adapting all of them is the standard
# "all-linear" QLoRA recipe.
QWEN_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LoRAConfig:
    """Hyper-parameters for the LoRA adapter."""

    r: int = field(default_factory=lambda: _env_int("FINAI_LORA_R", 16))
    lora_alpha: int = field(default_factory=lambda: _env_int("FINAI_LORA_ALPHA", 32))
    lora_dropout: float = field(default_factory=lambda: _env_float("FINAI_LORA_DROPOUT", 0.05))
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = QWEN_LORA_TARGET_MODULES

    def to_peft_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``peft.LoraConfig``."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": list(self.target_modules),
        }


@dataclass
class TrainingConfig:
    """End-to-end configuration for a single QLoRA training run."""

    # Identity ---------------------------------------------------------------
    run_name: str = field(default_factory=lambda: _env_str("FINAI_RUN_NAME", "finai-qlora"))

    # Dataset ----------------------------------------------------------------
    # ``dataset_path`` may point at a single file or a directory of JSONL files.
    dataset_path: str = field(default_factory=lambda: _env_str("FINAI_DATASET", "data/samples/alpaca_sample.jsonl"))
    dataset_format: str = field(default_factory=lambda: _env_str("FINAI_DATASET_FORMAT", "auto"))
    eval_split_ratio: float = field(default_factory=lambda: _env_float("FINAI_EVAL_RATIO", 0.05))
    max_seq_length: int = field(default_factory=lambda: _env_int("FINAI_MAX_SEQ_LEN", 2048))
    packing: bool = field(default_factory=lambda: _env_bool("FINAI_PACKING", False))

    # Optimisation -----------------------------------------------------------
    num_train_epochs: float = field(default_factory=lambda: _env_float("FINAI_EPOCHS", 3.0))
    # ``max_steps`` > 0 overrides ``num_train_epochs`` (useful for smoke tests).
    max_steps: int = field(default_factory=lambda: _env_int("FINAI_MAX_STEPS", -1))
    per_device_train_batch_size: int = field(default_factory=lambda: _env_int("FINAI_TRAIN_BS", 2))
    per_device_eval_batch_size: int = field(default_factory=lambda: _env_int("FINAI_EVAL_BS", 2))
    gradient_accumulation_steps: int = field(default_factory=lambda: _env_int("FINAI_GRAD_ACCUM", 8))
    learning_rate: float = field(default_factory=lambda: _env_float("FINAI_LR", 2e-4))
    lr_scheduler_type: str = field(default_factory=lambda: _env_str("FINAI_LR_SCHEDULER", "cosine"))
    warmup_ratio: float = field(default_factory=lambda: _env_float("FINAI_WARMUP_RATIO", 0.03))
    weight_decay: float = field(default_factory=lambda: _env_float("FINAI_WEIGHT_DECAY", 0.0))
    max_grad_norm: float = field(default_factory=lambda: _env_float("FINAI_MAX_GRAD_NORM", 1.0))
    optim: str = field(default_factory=lambda: _env_str("FINAI_OPTIM", "paged_adamw_8bit"))

    # Precision & memory -----------------------------------------------------
    # ``bf16`` is auto-resolved at runtime when left as ``None`` (see
    # :meth:`resolve_precision`); these explicit flags let callers force a mode.
    bf16: bool | None = None
    fp16: bool | None = None
    gradient_checkpointing: bool = field(default_factory=lambda: _env_bool("FINAI_GRAD_CKPT", True))

    # Logging / evaluation / checkpointing ----------------------------------
    logging_steps: int = field(default_factory=lambda: _env_int("FINAI_LOGGING_STEPS", 10))
    eval_strategy: str = field(default_factory=lambda: _env_str("FINAI_EVAL_STRATEGY", "steps"))
    eval_steps: int = field(default_factory=lambda: _env_int("FINAI_EVAL_STEPS", 100))
    save_strategy: str = field(default_factory=lambda: _env_str("FINAI_SAVE_STRATEGY", "steps"))
    save_steps: int = field(default_factory=lambda: _env_int("FINAI_SAVE_STEPS", 100))
    save_total_limit: int = field(default_factory=lambda: _env_int("FINAI_SAVE_TOTAL_LIMIT", 3))
    load_best_model_at_end: bool = field(default_factory=lambda: _env_bool("FINAI_LOAD_BEST", True))
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = field(default_factory=lambda: _env_str("FINAI_REPORT_TO", "tensorboard"))

    # Reproducibility --------------------------------------------------------
    seed: int = field(default_factory=lambda: _env_int("FINAI_SEED", 42))

    # Adapter ----------------------------------------------------------------
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def resolve_precision(self) -> tuple[bool, bool]:
        """Return the ``(bf16, fp16)`` pair to use, auto-detecting if unset.

        Prefers bf16 on supported GPUs (Ampere+), falls back to fp16 on older
        CUDA GPUs, and disables both on CPU. Explicit values always win.
        """
        if self.bf16 is not None or self.fp16 is not None:
            return bool(self.bf16), bool(self.fp16)

        from src.utils.env import has_cuda, supports_bf16

        if not has_cuda():
            return False, False
        if supports_bf16():
            return True, False
        return False, True

    def effective_batch_size(self, world_size: int = 1) -> int:
        """Global batch size = per-device * grad-accum * number of processes."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps * max(world_size, 1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot for logging / reproducibility."""
        bf16, fp16 = self.bf16, self.fp16
        return {
            "run_name": self.run_name,
            "dataset_path": self.dataset_path,
            "dataset_format": self.dataset_format,
            "eval_split_ratio": self.eval_split_ratio,
            "max_seq_length": self.max_seq_length,
            "packing": self.packing,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "optim": self.optim,
            "bf16": bf16,
            "fp16": fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "logging_steps": self.logging_steps,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "seed": self.seed,
            "lora": self.lora.to_peft_kwargs(),
        }


__all__ = ["TrainingConfig", "LoRAConfig", "QWEN_LORA_TARGET_MODULES"]
