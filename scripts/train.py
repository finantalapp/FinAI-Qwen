#!/usr/bin/env python
"""Fine-tune the model with QLoRA.

All settings default to :class:`config.training_config.TrainingConfig`; any flag
passed here overrides the corresponding field for this run only. Use ``--resume``
to continue from the latest checkpoint of a run with the same ``--run-name``.

Examples
--------
Quick smoke test (50 steps on the bundled sample data)::

    python scripts/train.py --max-steps 50 --eval-ratio 0

Full run on your own data::

    python scripts/train.py --dataset data/my_data.jsonl --epochs 3 --run-name finai-v1

Resume an interrupted run::

    python scripts/train.py --run-name finai-v1 --resume
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from src.training.trainer import run_training
from src.utils.bootstrap import setup
from src.utils.logging_utils import get_logger

logger = get_logger("finai.cli.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=None, help="Base model repo id to fine-tune.")
    parser.add_argument("--run-name", default=None, help="Run name (controls output/checkpoint/adapter dirs).")
    parser.add_argument("--dataset", default=None, help="Dataset file or directory.")
    parser.add_argument("--format", default=None, help="Dataset format (auto/alpaca/sharegpt/chatml/openai/prompt/text).")
    parser.add_argument("--epochs", type=float, default=None, help="Number of training epochs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Hard cap on training steps (overrides epochs).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max sequence length.")
    parser.add_argument("--eval-ratio", type=float, default=None, help="Eval split ratio (0 disables eval).")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Max checkpoints to keep.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--packing", action="store_true", help="Enable example packing.")
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint of this run.")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[TrainingConfig, ModelConfig]:
    """Apply CLI overrides on top of the default configs."""
    training_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    if args.model_id:
        model_cfg.model_id = args.model_id

    overrides = {
        "run_name": args.run_name,
        "dataset_path": args.dataset,
        "dataset_format": args.format,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_seq_length": args.max_seq_len,
        "eval_split_ratio": args.eval_ratio,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(training_cfg, field_name, value)
    if args.packing:
        training_cfg.packing = True

    if args.lora_r is not None:
        training_cfg.lora.r = args.lora_r
    if args.lora_alpha is not None:
        training_cfg.lora.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        training_cfg.lora.lora_dropout = args.lora_dropout

    return training_cfg, model_cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = setup()
    training_cfg, model_cfg = build_configs(args)

    adapter_dir = run_training(training_cfg, model_cfg, resume=args.resume, paths=paths)

    print("\nTraining finished.")
    print(f"  Adapter      : {adapter_dir}")
    print(f"  Checkpoints  : {paths.checkpoint_dir(training_cfg.run_name)}")
    print(f"  TensorBoard  : tensorboard --logdir {paths.tensorboard_dir(training_cfg.run_name)}")
    print(f"  Merge with   : python scripts/merge_lora.py --adapter {training_cfg.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
