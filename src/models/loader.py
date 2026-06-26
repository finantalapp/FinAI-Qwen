"""Load models and tokenizers, preferring the local (Drive) copy over the Hub.

A single :func:`load_model_and_tokenizer` entry point covers both inference and
training. It transparently:

* uses the downloaded copy under ``models/<name>`` when present, otherwise pulls
  from the Hub;
* applies 4-bit (QLoRA) quantisation when requested and a CUDA device exists;
* selects the right attention implementation (flash-attention-2 / SDPA);
* sets sane tokenizer padding for either generation (left) or training (right).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config.model_config import ModelConfig
from config.paths import ProjectPaths, configure_hf_cache, get_paths
from src.models.download import model_is_complete
from src.utils.env import has_cuda
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = get_logger("finai.model")


def resolve_source(model_cfg: ModelConfig, paths: ProjectPaths | None = None) -> tuple[str, bool]:
    """Return ``(source, is_local)`` for loading.

    Prefers the local downloaded directory; falls back to the Hub repo id.
    """
    paths = paths or get_paths()
    local = paths.model_dir(model_cfg.model_id)
    if model_is_complete(local):
        return str(local), True
    return model_cfg.model_id, False


def load_tokenizer(
    model_cfg: ModelConfig | None = None,
    *,
    source: str | None = None,
    padding_side: str = "right",
    paths: ProjectPaths | None = None,
) -> "PreTrainedTokenizerBase":
    """Load the tokenizer, ensuring a pad token and padding side are set."""
    from transformers import AutoTokenizer

    model_cfg = model_cfg or ModelConfig()
    paths = (paths or get_paths())
    configure_hf_cache(paths)

    is_local = source is not None
    if source is None:
        source, is_local = resolve_source(model_cfg, paths)

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=None if is_local else model_cfg.revision,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=True,
    )
    # Causal LMs need a pad token; fall back to EOS when none is defined.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_model(
    model_cfg: ModelConfig | None = None,
    *,
    source: str | None = None,
    for_training: bool = False,
    quantize: bool | None = None,
    paths: ProjectPaths | None = None,
) -> "PreTrainedModel":
    """Load the causal-LM weights with the configured precision/quantisation.

    Parameters
    ----------
    for_training:
        When ``True`` the model is prepared for k-bit training (input grads
        enabled, use_cache disabled) so a LoRA adapter can be attached.
    quantize:
        Force 4-bit quantisation on/off. ``None`` uses ``model_cfg.load_in_4bit``.
    """
    from transformers import AutoModelForCausalLM

    model_cfg = model_cfg or ModelConfig()
    paths = (paths or get_paths())
    configure_hf_cache(paths)

    is_local = source is not None
    if source is None:
        source, is_local = resolve_source(model_cfg, paths)
    logger.info("Loading model from %s (%s)", source, "local" if is_local else "hub")

    want_quant = model_cfg.load_in_4bit if quantize is None else quantize
    quant_config = None
    if want_quant:
        if has_cuda():
            quant_config = model_cfg.quantization_config()
        else:
            logger.warning("4-bit quantisation requested but no CUDA device found; loading in full precision.")

    attn = model_cfg.resolve_attn_implementation()
    logger.info("Attention implementation: %s | dtype: %s | quantised: %s", attn, model_cfg.dtype, quant_config is not None)

    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=None if is_local else model_cfg.revision,
        quantization_config=quant_config,
        torch_dtype=model_cfg.torch_dtype(),
        device_map=model_cfg.device_map if has_cuda() else None,
        attn_implementation=attn,
        trust_remote_code=model_cfg.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if for_training:
        # Disable the KV cache (incompatible with gradient checkpointing) and
        # make quantised inputs require grad so LoRA can back-propagate.
        model.config.use_cache = False
        if quant_config is not None:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
    else:
        model.eval()

    return model


def load_model_and_tokenizer(
    model_cfg: ModelConfig | None = None,
    *,
    for_training: bool = False,
    quantize: bool | None = None,
    padding_side: str | None = None,
    paths: ProjectPaths | None = None,
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    """Convenience wrapper returning a matched ``(model, tokenizer)`` pair."""
    model_cfg = model_cfg or ModelConfig()
    paths = paths or get_paths()
    source, _ = resolve_source(model_cfg, paths)

    if padding_side is None:
        padding_side = "right" if for_training else "left"

    tokenizer = load_tokenizer(model_cfg, source=source, padding_side=padding_side, paths=paths)
    model = load_model(
        model_cfg, source=source, for_training=for_training, quantize=quantize, paths=paths
    )
    return model, tokenizer


def load_plain_model(
    model_path: str | Path,
    model_cfg: ModelConfig | None = None,
    *,
    paths: ProjectPaths | None = None,
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    """Load a full-precision (non-quantised) model from an explicit path.

    Used for the merged model produced after LoRA training, and for benchmarking
    the un-quantised baseline.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = model_cfg or ModelConfig()
    paths = paths or get_paths()
    configure_hf_cache(paths)
    source = str(model_path)

    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, trust_remote_code=model_cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=model_cfg.torch_dtype(),
        device_map=model_cfg.device_map if has_cuda() else None,
        attn_implementation=model_cfg.resolve_attn_implementation(),
        trust_remote_code=model_cfg.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


__all__ = [
    "resolve_source",
    "load_tokenizer",
    "load_model",
    "load_model_and_tokenizer",
    "load_plain_model",
]
