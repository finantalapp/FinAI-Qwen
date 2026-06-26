#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base model to produce a standalone model.

The adapter can be given as a run name (resolved to ``adapters/<run>``) or as an
explicit directory path. The merged model is written to ``merged/<run>-merged``
unless ``--output`` is supplied.

Examples
--------
    python scripts/merge_lora.py --adapter finai-qlora
    python scripts/merge_lora.py --adapter "$FINAI_HOME/adapters/finai-v1" --device cpu
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig
from src.training.merge import merge_adapter
from src.utils.bootstrap import setup
from src.utils.logging_utils import get_logger

logger = get_logger("finai.cli.merge")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="Adapter run name or directory path.")
    parser.add_argument("--output", default=None, help="Output directory for the merged model.")
    parser.add_argument("--model-id", default=None, help="Base model repo id (must match what was trained).")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"], help="Where to perform the merge.")
    return parser.parse_args(argv)


def resolve_adapter(adapter_arg: str, paths) -> pathlib.Path:
    """Treat the argument as a path if it exists, otherwise as a run name."""
    candidate = pathlib.Path(adapter_arg)
    if candidate.is_dir():
        return candidate
    return paths.adapter_dir(adapter_arg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = setup()

    model_cfg = ModelConfig()
    if args.model_id:
        model_cfg.model_id = args.model_id

    adapter_dir = resolve_adapter(args.adapter, paths)
    if not adapter_dir.is_dir():
        logger.error("Adapter directory not found: %s", adapter_dir)
        return 1

    output = merge_adapter(adapter_dir, model_cfg, output_dir=args.output, device=args.device, paths=paths)
    print(f"\nMerged model written to: {output}")
    print(f"Serve it with: python chat.py --model-path {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
