#!/usr/bin/env python
"""Download the configured model from HuggingFace into Drive-backed storage.

Resumable, skips files that already match, and verifies the result. Safe to run
repeatedly: an already-complete model is detected instantly without hitting the
network.

Examples
--------
    python scripts/download_model.py
    python scripts/download_model.py --model-id Qwen/Qwen2.5-7B-Instruct
    python scripts/download_model.py --force --token hf_xxx
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig
from src.models.download import download_model
from src.models.verify import build_report
from src.utils.bootstrap import setup
from src.utils.logging_utils import get_logger

logger = get_logger("finai.cli.download")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=None, help="HuggingFace repo id (overrides config/FINAI_MODEL_ID).")
    parser.add_argument("--revision", default=None, help="Specific git revision / tag / commit to download.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the model looks complete.")
    parser.add_argument("--token", default=None, help="HuggingFace access token for gated/private models.")
    parser.add_argument("--no-report", action="store_true", help="Skip printing the verification report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup()

    model_cfg = ModelConfig()
    if args.model_id:
        model_cfg.model_id = args.model_id
    if args.revision:
        model_cfg.revision = args.revision

    path = download_model(model_cfg, force=args.force, token=args.token)
    logger.info("Model available at: %s", path)

    if not args.no_report:
        print(build_report(path).format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
