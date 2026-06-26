"""Model loading and quantisation configuration.

All values can be overridden through environment variables (handy on Colab,
where editing files is awkward) or by constructing :class:`ModelConfig`
explicitly. Heavy dependencies (``torch``, ``transformers``) are imported lazily
inside methods so this module stays importable in lightweight contexts such as
unit tests or the download script.

.. note::
   ``Qwen/Qwen2.5-8B`` does **not** exist on the HuggingFace Hub. The Qwen2.5
   family ships in 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B sizes (the ``8B``
   size only exists in the Qwen3 family). The default below therefore uses the
   closest real model, ``Qwen/Qwen2.5-7B-Instruct``. Override ``FINAI_MODEL_ID``
   to point at any other checkpoint, e.g. ``Qwen/Qwen3-8B``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from transformers import BitsAndBytesConfig

DType = Literal["bfloat16", "float16", "float32"]
AttnImpl = Literal["auto", "flash_attention_2", "sdpa", "eager"]

# The official default. Intentionally a real, existing checkpoint.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Files we actually need from the Hub. Excludes redundant formats (e.g. GGUF,
# original *.pth) to keep downloads lean and deterministic.
DEFAULT_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.json",
    "*.txt",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
    "*.tiktoken",
)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class ModelConfig:
    """Everything needed to download and load the model + tokenizer."""

    model_id: str = field(default_factory=lambda: _env_str("FINAI_MODEL_ID", DEFAULT_MODEL_ID))
    revision: str | None = field(default_factory=lambda: os.environ.get("FINAI_MODEL_REVISION") or None)

    # Compute precision used for the (non-quantised) parts of the model.
    dtype: DType = field(default_factory=lambda: _env_str("FINAI_DTYPE", "bfloat16"))  # type: ignore[assignment]

    # 4-bit (QLoRA-style) quantisation settings.
    load_in_4bit: bool = field(default_factory=lambda: _env_bool("FINAI_LOAD_IN_4BIT", True))
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: DType = field(
        default_factory=lambda: _env_str("FINAI_COMPUTE_DTYPE", "bfloat16")  # type: ignore[assignment]
    )

    # Attention kernel. ``auto`` resolves to flash-attention-2 when the package
    # is installed and the GPU supports it, otherwise PyTorch SDPA.
    attn_implementation: AttnImpl = field(
        default_factory=lambda: _env_str("FINAI_ATTN_IMPL", "auto")  # type: ignore[assignment]
    )

    device_map: str = field(default_factory=lambda: _env_str("FINAI_DEVICE_MAP", "auto"))
    trust_remote_code: bool = field(default_factory=lambda: _env_bool("FINAI_TRUST_REMOTE_CODE", False))

    # Maximum sequence length the tokenizer/model should assume.
    max_seq_length: int = field(default_factory=lambda: _env_int("FINAI_MAX_SEQ_LEN", 4096))

    # Patterns passed to ``snapshot_download``.
    allow_patterns: tuple[str, ...] = DEFAULT_ALLOW_PATTERNS

    # --- derived helpers ---------------------------------------------------
    @property
    def model_name(self) -> str:
        """The short model name (last path segment of :attr:`model_id`)."""
        return self.model_id.rstrip("/").split("/")[-1]

    def torch_dtype(self, which: DType | None = None) -> "torch.dtype":
        """Map a string dtype to a ``torch.dtype`` (torch imported lazily)."""
        import torch

        mapping: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[which or self.dtype]

    def resolve_attn_implementation(self) -> AttnImpl:
        """Resolve ``auto`` to a concrete attention implementation."""
        if self.attn_implementation != "auto":
            return self.attn_implementation

        # Local import to avoid a hard dependency at module import time.
        from src.utils.env import flash_attention_available

        return "flash_attention_2" if flash_attention_available() else "sdpa"

    def quantization_config(self) -> "BitsAndBytesConfig | None":
        """Build the bitsandbytes config, or ``None`` when 4-bit is disabled."""
        if not self.load_in_4bit:
            return None

        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self.torch_dtype(self.bnb_4bit_compute_dtype),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot, useful for logging and reproducibility."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "attn_implementation": self.attn_implementation,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "max_seq_length": self.max_seq_length,
        }


__all__ = ["ModelConfig", "DEFAULT_MODEL_ID", "DEFAULT_ALLOW_PATTERNS", "DType", "AttnImpl"]
