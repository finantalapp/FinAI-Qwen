"""Merge a trained LoRA adapter back into the base model.

The merged model is a standalone, full-precision checkpoint that can be served
without PEFT/bitsandbytes at inference time. Merging is done in fp16/bf16 (never
on the 4-bit quantised base, which would corrupt the weights); it can run on GPU
or, for memory-constrained runtimes, on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from config.model_config import ModelConfig
from config.paths import ProjectPaths, configure_hf_cache, get_paths
from src.utils.env import has_cuda
from src.utils.logging_utils import get_logger

logger = get_logger("finai.merge")


def _infer_run_name(adapter_dir: Path) -> str:
    return adapter_dir.name


def merge_adapter(
    adapter_dir: Path | str,
    model_cfg: ModelConfig | None = None,
    *,
    output_dir: Path | str | None = None,
    device: str | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    """Merge the adapter at ``adapter_dir`` into the base model.

    Parameters
    ----------
    adapter_dir:
        Directory holding the trained LoRA adapter.
    output_dir:
        Where to write the merged model. Defaults to ``merged/<run>-merged``.
    device:
        ``"auto"``/``"cuda"`` to merge on GPU, ``"cpu"`` to merge in host RAM
        (slower but works on small GPUs). Defaults to GPU when available.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.models.loader import resolve_source

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model_cfg = model_cfg or ModelConfig()
    paths = (paths or get_paths()).ensure()
    configure_hf_cache(paths)

    run_name = _infer_run_name(adapter_dir)
    output_dir = Path(output_dir) if output_dir else paths.merged_dir(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("auto" if has_cuda() else "cpu")
    device_map = {"": "cpu"} if device == "cpu" else device
    base_source, is_local = resolve_source(model_cfg, paths)
    logger.info("Loading base model '%s' (%s) on %s for merge", base_source, "local" if is_local else "hub", device)

    base = AutoModelForCausalLM.from_pretrained(
        base_source,
        torch_dtype=model_cfg.torch_dtype(),
        device_map=device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    logger.info("Applying adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

    logger.info("Merging adapter into base weights ...")
    merged = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s", output_dir)
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    # Save the tokenizer alongside the merged weights so the directory is a
    # complete, self-contained model.
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    tokenizer.save_pretrained(str(output_dir))

    # Record provenance.
    (output_dir / "finai_merge_info.json").write_text(
        json.dumps(
            {
                "base_model": model_cfg.model_id,
                "adapter_dir": str(adapter_dir),
                "merged_dir": str(output_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Merge complete: %s", output_dir)
    return output_dir


__all__ = ["merge_adapter"]
