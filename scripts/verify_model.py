#!/usr/bin/env python
"""Print a detailed report about a downloaded model.

Reports parameter count, dtype breakdown, on-disk size, per-file sizes, file
integrity, tokenizer class and generation config - all by reading file headers,
so it is fast and needs no GPU.

Examples
--------
    python scripts/verify_model.py
    python scripts/verify_model.py --model-id Qwen/Qwen2.5-7B-Instruct
    python scripts/verify_model.py --path "$FINAI_HOME/merged/finai-qlora-merged"
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config.model_config import ModelConfig
from config.paths import get_paths
from src.models.verify import build_report
from src.utils.bootstrap import setup


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=None, help="Explicit model directory to inspect.")
    parser.add_argument("--model-id", default=None, help="Locate the model directory by HuggingFace repo id.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = setup()

    if args.path:
        target = pathlib.Path(args.path)
    else:
        model_cfg = ModelConfig()
        if args.model_id:
            model_cfg.model_id = args.model_id
        target = paths.model_dir(model_cfg.model_id)

    report = build_report(target)
    print(report.format())
    return 0 if report.integrity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
