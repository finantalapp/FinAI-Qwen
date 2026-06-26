"""Download model weights from the HuggingFace Hub to (Drive-backed) storage.

Built on ``huggingface_hub.snapshot_download``, which already provides:

* resumable downloads (interrupted transfers continue from where they stopped),
* content-addressed skipping (files whose hash already matches are not
  re-downloaded), and
* a progress bar.

On top of that this module adds a completeness check so an already-downloaded
model is detected instantly without contacting the Hub, and a post-download
verification of the safetensors shard set.
"""

from __future__ import annotations

import json
from pathlib import Path

from config.model_config import ModelConfig
from config.paths import ProjectPaths, configure_hf_cache, get_paths
from src.utils.logging_utils import get_logger

logger = get_logger("finai.download")

# Minimum set of non-weight files a usable model directory must contain.
_REQUIRED_META_FILES = ("config.json",)
_TOKENIZER_FILES_ANY = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
_SAFETENSORS_INDEX = "model.safetensors.index.json"


def _safetensors_shards_complete(local_dir: Path) -> bool:
    """Return ``True`` when all expected safetensors shards are present."""
    index = local_dir / _SAFETENSORS_INDEX
    if index.exists():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        except (json.JSONDecodeError, OSError):
            return False
        shards = set(weight_map.values())
        return bool(shards) and all((local_dir / shard).exists() for shard in shards)
    # No index => single-shard model; require at least one safetensors file.
    return any(local_dir.glob("*.safetensors"))


def model_is_complete(local_dir: Path | str) -> bool:
    """Heuristically determine whether a local directory holds a usable model."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        return False
    if not all((local_dir / name).exists() for name in _REQUIRED_META_FILES):
        return False
    if not any((local_dir / name).exists() for name in _TOKENIZER_FILES_ANY):
        return False
    return _safetensors_shards_complete(local_dir)


def verify_download(local_dir: Path | str) -> list[str]:
    """Return a list of problems with a downloaded model (empty == healthy)."""
    local_dir = Path(local_dir)
    problems: list[str] = []
    if not local_dir.is_dir():
        return [f"Directory does not exist: {local_dir}"]
    for name in _REQUIRED_META_FILES:
        if not (local_dir / name).exists():
            problems.append(f"Missing required file: {name}")
    if not any((local_dir / name).exists() for name in _TOKENIZER_FILES_ANY):
        problems.append("No tokenizer files found.")
    if not _safetensors_shards_complete(local_dir):
        problems.append("Safetensors weight shards are missing or incomplete.")
    return problems


def download_model(
    model_cfg: ModelConfig | None = None,
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Download the configured model to its local directory and return the path.

    Parameters
    ----------
    model_cfg:
        The model configuration (defaults to a fresh :class:`ModelConfig`).
    paths:
        Project paths (defaults to the resolved singleton).
    force:
        Re-download even if the model already looks complete.
    token:
        Optional HuggingFace access token for gated/private models.
    """
    model_cfg = model_cfg or ModelConfig()
    paths = (paths or get_paths()).ensure()
    configure_hf_cache(paths)

    target = paths.model_dir(model_cfg.model_id)
    target.mkdir(parents=True, exist_ok=True)

    if not force and model_is_complete(target):
        logger.info("Model already present and complete at %s - skipping download.", target)
        return target

    from huggingface_hub import snapshot_download

    logger.info(
        "Downloading '%s'%s to %s",
        model_cfg.model_id,
        f" (revision {model_cfg.revision})" if model_cfg.revision else "",
        target,
    )
    snapshot_download(
        repo_id=model_cfg.model_id,
        revision=model_cfg.revision,
        local_dir=str(target),
        allow_patterns=list(model_cfg.allow_patterns),
        token=token,
    )

    problems = verify_download(target)
    if problems:
        raise RuntimeError(
            "Download finished but verification failed:\n  - " + "\n  - ".join(problems)
        )
    logger.info("Download verified successfully at %s", target)
    return target


__all__ = ["download_model", "model_is_complete", "verify_download"]
