#!/usr/bin/env python
"""Benchmark generation speed and resource usage, then print a full report.

Measures, averaged over several runs (after warm-up):

* total latency per request,
* time-to-first-token (TTFT),
* overall and steady-state throughput (tokens/sec),
* peak VRAM and host RAM usage,
* model load time.

Greedy decoding (temperature 0) is used by default so timings are stable and
comparable across runs.

Examples
--------
    python scripts/benchmark.py
    python scripts/benchmark.py --model-path "$FINAI_HOME/merged/finai-qlora-merged" --runs 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from time import perf_counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config.generation_config import GenerationSettings
from config.model_config import ModelConfig
from src.inference.engine import InferenceEngine
from src.utils.bootstrap import setup
from src.utils.env import has_cuda
from src.utils.logging_utils import get_logger
from src.utils.system import (
    RAMStats,
    VRAMStats,
    format_system_report,
    ram_usage,
    reset_peak_memory,
    vram_usage,
)

logger = get_logger("finai.cli.benchmark")

DEFAULT_PROMPT = "Explain the concept of compound interest and give a worked example."


@dataclass
class BenchmarkResult:
    """Aggregated benchmark metrics."""

    runs: int
    prompt_tokens: int
    gen_tokens: int
    load_time_s: float
    avg_total_s: float
    avg_ttft_s: float
    overall_tps: float
    decode_tps: float
    vram: VRAMStats
    ram: RAMStats

    def format(self) -> str:
        lines = [
            "=" * 64,
            "Benchmark report",
            "=" * 64,
            f"Runs                  : {self.runs}",
            f"Prompt tokens         : {self.prompt_tokens}",
            f"Generated tokens      : {self.gen_tokens}",
            f"Model load time       : {self.load_time_s:.2f} s",
            f"Avg total latency     : {self.avg_total_s:.3f} s",
            f"Avg time-to-first-tok : {self.avg_ttft_s * 1000:.1f} ms",
            f"Throughput (overall)  : {self.overall_tps:.1f} tokens/sec",
            f"Throughput (decode)   : {self.decode_tps:.1f} tokens/sec",
            "",
            format_system_report(),
            "=" * 64,
        ]
        return "\n".join(lines)


def _generation_kwargs(engine: InferenceEngine, settings: GenerationSettings) -> dict:
    kwargs = settings.to_kwargs()
    kwargs.setdefault("pad_token_id", engine.tokenizer.pad_token_id)
    if engine.tokenizer.eos_token_id is not None:
        kwargs.setdefault("eos_token_id", engine.tokenizer.eos_token_id)
    return kwargs


def _sync() -> None:
    if has_cuda():
        import torch

        torch.cuda.synchronize()


def timed_generate(engine: InferenceEngine, messages, settings) -> tuple[int, int, float]:
    """Run one full generation; return ``(prompt_tokens, gen_tokens, seconds)``."""
    import torch

    inputs = engine.build_inputs(messages, settings.system_prompt)
    prompt_len = int(inputs["input_ids"].shape[1])

    _sync()
    start = perf_counter()
    with torch.inference_mode():
        output = engine.model.generate(**inputs, **_generation_kwargs(engine, settings))
    _sync()
    elapsed = perf_counter() - start

    gen_tokens = int(output.shape[1]) - prompt_len
    return prompt_len, gen_tokens, elapsed


def measure_ttft(engine: InferenceEngine, messages, settings) -> float:
    """Return seconds until the first token is produced."""
    # Cap new tokens so the background generation thread finishes quickly; TTFT
    # itself is independent of ``max_new_tokens``.
    probe = settings.with_overrides(max_new_tokens=min(settings.max_new_tokens, 8))
    start = perf_counter()
    for _ in engine.stream(messages, probe):
        return perf_counter() - start
    return perf_counter() - start


def run_benchmark(
    engine: InferenceEngine,
    settings: GenerationSettings,
    *,
    prompt: str,
    runs: int,
    warmup: int,
    load_time_s: float,
) -> BenchmarkResult:
    messages = [{"role": "user", "content": prompt}]

    for _ in range(max(warmup, 0)):
        timed_generate(engine, messages, settings)

    totals: list[float] = []
    ttfts: list[float] = []
    prompt_tokens = gen_tokens = 0
    reset_peak_memory()
    for index in range(runs):
        prompt_tokens, gen_tokens, elapsed = timed_generate(engine, messages, settings)
        ttft = measure_ttft(engine, messages, settings)
        totals.append(elapsed)
        ttfts.append(ttft)
        logger.info("Run %d/%d: %.3fs total, %d tokens, TTFT %.0fms", index + 1, runs, elapsed, gen_tokens, ttft * 1000)

    avg_total = sum(totals) / len(totals)
    avg_ttft = sum(ttfts) / len(ttfts)
    overall_tps = gen_tokens / avg_total if avg_total > 0 else 0.0
    decode_tps = (gen_tokens - 1) / (avg_total - avg_ttft) if avg_total > avg_ttft and gen_tokens > 1 else overall_tps

    return BenchmarkResult(
        runs=runs,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        load_time_s=load_time_s,
        avg_total_s=avg_total,
        avg_ttft_s=avg_ttft,
        overall_tps=overall_tps,
        decode_tps=decode_tps,
        vram=vram_usage(),
        ram=ram_usage(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default=None, help="Explicit (e.g. merged) model directory.")
    parser.add_argument("--model-id", default=None, help="Override the model repo id.")
    parser.add_argument("--no-quantize", action="store_true", help="Benchmark full precision instead of 4-bit.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to generate from.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Tokens to generate per run.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--runs", type=int, default=3, help="Number of measured runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up runs (not measured).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup()

    model_cfg = ModelConfig()
    if args.model_id:
        model_cfg.model_id = args.model_id

    load_start = perf_counter()
    if args.model_path:
        engine = InferenceEngine.from_path(args.model_path, model_cfg)
    else:
        engine = InferenceEngine.from_config(model_cfg, quantize=False if args.no_quantize else None)
    load_time = perf_counter() - load_start

    settings = GenerationSettings(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    result = run_benchmark(
        engine, settings, prompt=args.prompt, runs=args.runs, warmup=args.warmup, load_time_s=load_time
    )
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
