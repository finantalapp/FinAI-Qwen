"""Quantitative and qualitative evaluation of a model.

* :func:`compute_perplexity` - token-level perplexity over a set of texts, a
  standard intrinsic measure of language-model quality (lower is better).
* :func:`sample_generations` - run a handful of prompts through the model for a
  quick qualitative smell-test.

Both are tied together by :func:`evaluate`, which loads the model, prepares the
eval texts from any supported dataset format, and returns an :class:`EvalReport`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config.generation_config import GenerationSettings
from config.model_config import ModelConfig
from config.paths import ProjectPaths, get_paths
from src.inference.engine import InferenceEngine
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = get_logger("finai.eval")

DEFAULT_PROMPTS: tuple[str, ...] = (
    "In one paragraph, explain what compound interest is and why it matters.",
    "What is the difference between a stock and a bond?",
    "Give three practical tips for building an emergency fund.",
)


@dataclass
class EvalReport:
    """Container for evaluation results."""

    num_perplexity_samples: int = 0
    perplexity: float | None = None
    sample_outputs: list[tuple[str, str]] = field(default_factory=list)

    def format(self) -> str:
        lines = ["=" * 64, "Evaluation report", "=" * 64]
        if self.perplexity is not None:
            lines.append(f"Perplexity ({self.num_perplexity_samples} samples): {self.perplexity:.3f}")
        else:
            lines.append("Perplexity: (not computed)")
        if self.sample_outputs:
            lines.append("")
            lines.append("Sample generations:")
            for i, (prompt, output) in enumerate(self.sample_outputs, start=1):
                lines.append(f"\n[{i}] Prompt: {prompt}")
                lines.append(f"    Output: {output.strip()}")
        lines.append("=" * 64)
        return "\n".join(lines)


def compute_perplexity(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    texts: list[str],
    *,
    max_length: int = 1024,
) -> tuple[float, int]:
    """Compute corpus perplexity over ``texts`` (returns ``(ppl, n_used)``).

    Uses a single forward pass per example with the labels equal to the inputs,
    weighting each example's mean loss by its token count so the final figure is
    a proper token-level perplexity.
    """
    import torch

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    used = 0
    for text in texts:
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc.input_ids.to(model.device)
        if input_ids.shape[1] < 2:
            continue
        with torch.inference_mode():
            loss = model(input_ids, labels=input_ids).loss
        n_tokens = input_ids.shape[1] - 1  # labels shift by one
        total_nll += float(loss) * n_tokens
        total_tokens += n_tokens
        used += 1

    if total_tokens == 0:
        return float("nan"), 0
    return math.exp(total_nll / total_tokens), used


def sample_generations(
    engine: InferenceEngine,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    settings: GenerationSettings | None = None,
) -> list[tuple[str, str]]:
    """Generate a completion for each prompt; return ``(prompt, output)`` pairs."""
    settings = settings or GenerationSettings()
    results: list[tuple[str, str]] = []
    for prompt in prompts:
        output = engine.generate([{"role": "user", "content": prompt}], settings)
        results.append((prompt, output))
    return results


def evaluate(
    *,
    model_path: str | None = None,
    model_cfg: ModelConfig | None = None,
    dataset_path: str | None = None,
    max_samples: int = 200,
    max_length: int = 1024,
    run_samples: bool = True,
    paths: ProjectPaths | None = None,
) -> EvalReport:
    """Load a model and produce an :class:`EvalReport`.

    Parameters
    ----------
    model_path:
        Explicit (e.g. merged) model directory. When ``None`` the configured
        base model is used.
    dataset_path:
        Dataset to compute perplexity over. When ``None`` perplexity is skipped.
    """
    model_cfg = model_cfg or ModelConfig()
    paths = paths or get_paths()

    if model_path:
        engine = InferenceEngine.from_path(model_path, model_cfg)
    else:
        engine = InferenceEngine.from_config(model_cfg, paths=paths)

    report = EvalReport()

    if dataset_path:
        from src.data.loader import prepare_dataset

        bundle = prepare_dataset(
            dataset_path, engine.tokenizer, eval_split_ratio=0.0, paths=paths
        )
        texts = bundle.train["text"][:max_samples]
        logger.info("Computing perplexity over %d examples", len(texts))
        ppl, used = compute_perplexity(engine.model, engine.tokenizer, list(texts), max_length=max_length)
        report.perplexity = ppl
        report.num_perplexity_samples = used

    if run_samples:
        logger.info("Generating qualitative samples")
        report.sample_outputs = sample_generations(engine)

    return report


__all__ = [
    "EvalReport",
    "compute_perplexity",
    "sample_generations",
    "evaluate",
    "DEFAULT_PROMPTS",
]
