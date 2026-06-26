#!/usr/bin/env python
"""Evaluate a model: perplexity over a dataset plus qualitative sample outputs.

Examples
--------
Evaluate the base model qualitatively only::

    python scripts/evaluate.py

Compute perplexity on a dataset and show samples::

    python scripts/evaluate.py --dataset data/samples/openai_sample.jsonl

Evaluate a merged fine-tuned model::

    python scripts/evaluate.py --model-path "$FINAI_HOME/merged/finai-qlora-merged" \\
        --dataset data/samples/openai_sample.jsonl
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig
from src.evaluation.evaluator import evaluate
from src.utils.bootstrap import setup


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default=None, help="Explicit (e.g. merged) model directory.")
    parser.add_argument("--model-id", default=None, help="Override the model repo id.")
    parser.add_argument("--dataset", default=None, help="Dataset for perplexity (omit to skip perplexity).")
    parser.add_argument("--max-samples", type=int, default=200, help="Max examples used for perplexity.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max tokens per perplexity example.")
    parser.add_argument("--no-samples", action="store_true", help="Skip qualitative sample generations.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = setup()

    model_cfg = ModelConfig()
    if args.model_id:
        model_cfg.model_id = args.model_id

    report = evaluate(
        model_path=args.model_path,
        model_cfg=model_cfg,
        dataset_path=args.dataset,
        max_samples=args.max_samples,
        max_length=args.max_length,
        run_samples=not args.no_samples,
        paths=paths,
    )
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
