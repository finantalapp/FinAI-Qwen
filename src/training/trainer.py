"""QLoRA fine-tuning built on TRL's :class:`SFTTrainer`.

The pipeline:

1. Load the base model 4-bit-quantised and prepared for k-bit training.
2. Attach a LoRA adapter (PEFT) to the configured target modules.
3. Render the dataset to a single ``text`` column via the chat template.
4. Train with gradient accumulation/checkpointing, bf16 (auto-detected),
   periodic evaluation, TensorBoard logging and bounded checkpoint retention.
5. Save the final adapter (+ tokenizer + config snapshot) to ``adapters/<run>``.

TRL/Transformers rename constructor arguments between releases, so config and
trainer kwargs are filtered against the *installed* signatures (see
:func:`_supported_fields` / :func:`_adapt_config_kwargs`). This keeps the module
working across a range of "latest stable" versions instead of pinning to one.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.model_config import ModelConfig
from config.paths import ProjectPaths, get_paths
from config.training_config import TrainingConfig
from src.data.loader import DatasetBundle, prepare_dataset
from src.utils.env import set_seed
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = get_logger("finai.train")

# Argument renames across TRL/Transformers versions (new name -> old aliases).
_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "eval_strategy": ("evaluation_strategy",),
    "max_seq_length": ("max_length",),
}


def _supported_fields(cls: type) -> set[str]:
    """Return the set of constructor field names a class accepts."""
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(inspect.signature(cls.__init__).parameters)


def _adapt_config_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown kwargs and remap renamed ones to fit ``cls``."""
    fields = _supported_fields(cls)
    adapted: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in fields:
            adapted[key] = value
            continue
        for alias in _CONFIG_ALIASES.get(key, ()):  # try known older names
            if alias in fields:
                adapted[alias] = value
                break
        else:
            logger.debug("Ignoring unsupported config argument: %s", key)
    return adapted


def build_sft_config(training_cfg: TrainingConfig, paths: ProjectPaths):
    """Build a TRL ``SFTConfig`` from :class:`TrainingConfig`."""
    from trl import SFTConfig

    bf16, fp16 = training_cfg.resolve_precision()
    output_dir = paths.checkpoint_dir(training_cfg.run_name)
    logging_dir = paths.tensorboard_dir(training_cfg.run_name)
    report_to = [] if training_cfg.report_to.lower() == "none" else [training_cfg.report_to]
    has_eval = training_cfg.eval_split_ratio > 0

    desired: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": training_cfg.run_name,
        "num_train_epochs": training_cfg.num_train_epochs,
        "max_steps": training_cfg.max_steps,
        "per_device_train_batch_size": training_cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": training_cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
        "learning_rate": training_cfg.learning_rate,
        "lr_scheduler_type": training_cfg.lr_scheduler_type,
        "warmup_ratio": training_cfg.warmup_ratio,
        "weight_decay": training_cfg.weight_decay,
        "max_grad_norm": training_cfg.max_grad_norm,
        "optim": training_cfg.optim,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": training_cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": training_cfg.logging_steps,
        "logging_dir": str(logging_dir),
        "eval_strategy": training_cfg.eval_strategy if has_eval else "no",
        "eval_steps": training_cfg.eval_steps,
        "save_strategy": training_cfg.save_strategy,
        "save_steps": training_cfg.save_steps,
        "save_total_limit": training_cfg.save_total_limit,
        "load_best_model_at_end": training_cfg.load_best_model_at_end and has_eval,
        "metric_for_best_model": training_cfg.metric_for_best_model,
        "greater_is_better": training_cfg.greater_is_better,
        "report_to": report_to,
        "seed": training_cfg.seed,
        "max_seq_length": training_cfg.max_seq_length,
        "packing": training_cfg.packing,
        "dataset_text_field": "text",
        "dataloader_pin_memory": True,
        "group_by_length": True,
    }
    return SFTConfig(**_adapt_config_kwargs(SFTConfig, desired))


def attach_lora(model: "PreTrainedModel", training_cfg: TrainingConfig) -> "PreTrainedModel":
    """Wrap ``model`` with a LoRA adapter and log the trainable-parameter ratio."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(**training_cfg.lora.to_peft_kwargs())
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA attached: %s trainable / %s total params (%.4f%%)",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1),
    )
    return peft_model


def build_trainer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    data: DatasetBundle,
    training_cfg: TrainingConfig,
    paths: ProjectPaths,
):
    """Construct a TRL ``SFTTrainer`` for the given model and dataset."""
    from trl import SFTTrainer

    sft_config = build_sft_config(training_cfg, paths)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": data.train,
    }
    if data.eval is not None:
        trainer_kwargs["eval_dataset"] = data.eval

    # ``tokenizer`` was renamed to ``processing_class`` in newer TRL releases.
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer

    return SFTTrainer(**trainer_kwargs)


def _find_resume_checkpoint(output_dir: Path) -> str | None:
    """Return the latest checkpoint path under ``output_dir`` (or ``None``)."""
    if not output_dir.is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(output_dir))
    return last


def run_training(
    training_cfg: TrainingConfig | None = None,
    model_cfg: ModelConfig | None = None,
    *,
    resume: bool = False,
    paths: ProjectPaths | None = None,
) -> Path:
    """Run an end-to-end QLoRA fine-tune and return the saved adapter path."""
    from src.models.loader import load_model_and_tokenizer

    training_cfg = training_cfg or TrainingConfig()
    model_cfg = model_cfg or ModelConfig()
    paths = (paths or get_paths()).ensure()

    set_seed(training_cfg.seed)
    logger.info("Run '%s' | effective batch size: %d", training_cfg.run_name, training_cfg.effective_batch_size())

    model, tokenizer = load_model_and_tokenizer(
        model_cfg, for_training=True, quantize=True, padding_side="right", paths=paths
    )
    if not training_cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    model = attach_lora(model, training_cfg)

    data = prepare_dataset(
        training_cfg.dataset_path,
        tokenizer,
        fmt=training_cfg.dataset_format,  # type: ignore[arg-type]
        eval_split_ratio=training_cfg.eval_split_ratio,
        seed=training_cfg.seed,
        paths=paths,
    )
    logger.info("Dataset ready: %d train / %d eval examples", data.num_train, data.num_eval)

    trainer = build_trainer(model, tokenizer, data, training_cfg, paths)

    resume_from = None
    if resume:
        resume_from = _find_resume_checkpoint(paths.checkpoint_dir(training_cfg.run_name))
        if resume_from:
            logger.info("Resuming from checkpoint: %s", resume_from)
        else:
            logger.warning("Resume requested but no checkpoint found; starting fresh.")

    trainer.train(resume_from_checkpoint=resume_from)

    # Persist the final adapter, tokenizer and a config snapshot.
    adapter_dir = paths.adapter_dir(training_cfg.run_name)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    snapshot = {"model": model_cfg.to_dict(), "training": training_cfg.to_dict()}
    (adapter_dir / "finai_run_config.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    logger.info("Training complete. Adapter saved to %s", adapter_dir)
    return adapter_dir


__all__ = [
    "build_sft_config",
    "attach_lora",
    "build_trainer",
    "run_training",
]
