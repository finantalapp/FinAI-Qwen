"""Text-generation (sampling) configuration.

These settings drive both the chat UI and the benchmark/evaluation scripts.
Every field maps cleanly onto ``model.generate`` keyword arguments via
:meth:`GenerationSettings.to_kwargs`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are FinAI, a helpful, accurate and concise financial-domain "
    "assistant. Answer clearly, show your reasoning when it helps, and say "
    "when you are unsure instead of inventing facts."
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class GenerationSettings:
    """Sampling parameters for a single generation request."""

    max_new_tokens: int = field(default_factory=lambda: _env_int("FINAI_MAX_NEW_TOKENS", 1024))
    temperature: float = field(default_factory=lambda: _env_float("FINAI_TEMPERATURE", 0.7))
    top_p: float = field(default_factory=lambda: _env_float("FINAI_TOP_P", 0.8))
    top_k: int = field(default_factory=lambda: _env_int("FINAI_TOP_K", 20))
    repetition_penalty: float = field(default_factory=lambda: _env_float("FINAI_REPETITION_PENALTY", 1.05))
    do_sample: bool = True
    system_prompt: str = field(
        default_factory=lambda: os.environ.get("FINAI_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    )

    def __post_init__(self) -> None:
        # A temperature of 0 is a common way to ask for greedy decoding; honour
        # it by disabling sampling rather than dividing by zero downstream.
        if self.temperature <= 0:
            self.do_sample = False

    def to_kwargs(self) -> dict[str, Any]:
        """Return the subset of ``generate`` kwargs that this object controls.

        When ``do_sample`` is ``False`` the sampling-only knobs are omitted so
        HuggingFace does not emit "ignored sampling parameter" warnings.
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        return kwargs

    def with_overrides(self, **overrides: Any) -> "GenerationSettings":
        """Return a copy with selected fields overridden (e.g. from a UI)."""
        from dataclasses import replace

        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)


__all__ = ["GenerationSettings", "DEFAULT_SYSTEM_PROMPT"]
