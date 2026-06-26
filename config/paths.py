"""Centralised filesystem layout for FinAI-Qwen.

This module is the *single source of truth* for every path the project uses.
No other module is allowed to hard-code a directory or filename; everything is
derived from :class:`ProjectPaths`, which in turn adapts automatically to the
runtime environment (local workstation, Google Colab with Drive, or Colab
without Drive).

Resolution order for the artefact root (``FINAI_HOME``):

1. The ``FINAI_HOME`` environment variable, if set (highest priority).
2. ``<Drive>/FinAI-Qwen`` when running on Colab with Google Drive mounted.
3. ``/content/FinAI-Qwen`` when on Colab *without* Drive (ephemeral, warned).
4. The repository root when running on a normal workstation.

The module deliberately depends only on the standard library so it can be
imported in any context (including unit tests) without pulling in torch or
transformers.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

# Repository root: ``config/`` lives one level below the project root.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# Conventional Google Drive mount point on Colab.
_DRIVE_MOUNT: Final[Path] = Path("/content/drive/MyDrive")

# Environment variable that overrides the artefact root.
ENV_HOME: Final[str] = "FINAI_HOME"


def _running_on_colab() -> bool:
    """Return ``True`` when executing inside a Google Colab runtime."""
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def _drive_is_mounted() -> bool:
    """Return ``True`` when Google Drive is mounted at the standard location."""
    return _DRIVE_MOUNT.exists()


def _resolve_home() -> Path:
    """Resolve the root directory under which all artefacts are stored."""
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser().resolve()

    if _running_on_colab():
        if _drive_is_mounted():
            return _DRIVE_MOUNT / "FinAI-Qwen"
        # Drive not mounted yet: fall back to ephemeral storage. Callers are
        # warned via :meth:`ProjectPaths.warnings`.
        return Path("/content/FinAI-Qwen")

    return REPO_ROOT


def sanitize_model_id(model_id: str) -> str:
    """Turn a HuggingFace repo id into a filesystem-safe directory name.

    ``"Qwen/Qwen2.5-7B-Instruct"`` -> ``"Qwen2.5-7B-Instruct"``.
    """
    name = model_id.strip().rstrip("/").split("/")[-1]
    return name.replace(os.sep, "_").replace(" ", "_")


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable view of every directory used by the project.

    Instances are cheap value objects. Call :meth:`ensure` to materialise the
    base directories on disk before writing to them.
    """

    home: Path
    repo_root: Path = REPO_ROOT
    is_colab: bool = False
    drive_mounted: bool = False

    # --- base directories (all relative to ``home``) -----------------------
    models_root: Path = field(init=False)
    datasets_root: Path = field(init=False)
    checkpoints_root: Path = field(init=False)
    adapters_root: Path = field(init=False)
    merged_root: Path = field(init=False)
    logs_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)  # TensorBoard event files
    cache_dir: Path = field(init=False)  # HuggingFace / datasets cache

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        # ``frozen=True`` forbids normal attribute assignment, so we go through
        # ``object.__setattr__`` to populate the derived fields exactly once.
        object.__setattr__(self, "models_root", self.home / "models")
        object.__setattr__(self, "datasets_root", self.home / "datasets")
        object.__setattr__(self, "checkpoints_root", self.home / "checkpoints")
        object.__setattr__(self, "adapters_root", self.home / "adapters")
        object.__setattr__(self, "merged_root", self.home / "merged")
        object.__setattr__(self, "logs_dir", self.home / "logs")
        object.__setattr__(self, "outputs_dir", self.home / "outputs")
        object.__setattr__(self, "runs_dir", self.home / "runs")
        object.__setattr__(self, "cache_dir", self.home / "cache")

    # --- derived, parameterised locations ----------------------------------
    def model_dir(self, model_id: str) -> Path:
        """Local directory holding the downloaded base model weights."""
        return self.models_root / sanitize_model_id(model_id)

    def checkpoint_dir(self, run_name: str) -> Path:
        """Directory where a training run writes its checkpoints."""
        return self.checkpoints_root / run_name

    def adapter_dir(self, run_name: str) -> Path:
        """Directory where the final LoRA adapter of a run is stored."""
        return self.adapters_root / run_name

    def merged_dir(self, run_name: str) -> Path:
        """Directory where a merged (adapter + base) model is stored."""
        return self.merged_root / f"{run_name}-merged"

    def tensorboard_dir(self, run_name: str) -> Path:
        """Per-run TensorBoard log directory."""
        return self.runs_dir / run_name

    # --- lifecycle ---------------------------------------------------------
    @property
    def base_dirs(self) -> tuple[Path, ...]:
        """The base directories created by :meth:`ensure`."""
        return (
            self.models_root,
            self.datasets_root,
            self.checkpoints_root,
            self.adapters_root,
            self.merged_root,
            self.logs_dir,
            self.outputs_dir,
            self.runs_dir,
            self.cache_dir,
        )

    def ensure(self) -> "ProjectPaths":
        """Create all base directories if they do not already exist."""
        for directory in self.base_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def warnings(self) -> list[str]:
        """Return human-readable warnings about the current configuration."""
        messages: list[str] = []
        if self.is_colab and not self.drive_mounted and ENV_HOME not in os.environ:
            messages.append(
                "Running on Colab without Google Drive mounted. Artefacts will "
                "be written to ephemeral storage and lost when the runtime "
                "recycles. Mount Drive or set FINAI_HOME to persist them."
            )
        return messages

    def as_dict(self) -> dict[str, str]:
        """Flat mapping of path names to their string values (for logging)."""
        return {
            "home": str(self.home),
            "repo_root": str(self.repo_root),
            "models_root": str(self.models_root),
            "datasets_root": str(self.datasets_root),
            "checkpoints_root": str(self.checkpoints_root),
            "adapters_root": str(self.adapters_root),
            "merged_root": str(self.merged_root),
            "logs_dir": str(self.logs_dir),
            "outputs_dir": str(self.outputs_dir),
            "runs_dir": str(self.runs_dir),
            "cache_dir": str(self.cache_dir),
        }


@lru_cache(maxsize=1)
def get_paths() -> ProjectPaths:
    """Return the process-wide :class:`ProjectPaths` singleton.

    The result is cached so repeated calls are free and always agree. The cache
    can be cleared in tests via ``get_paths.cache_clear()``.
    """
    return ProjectPaths(
        home=_resolve_home(),
        repo_root=REPO_ROOT,
        is_colab=_running_on_colab(),
        drive_mounted=_drive_is_mounted(),
    )


def configure_hf_cache(paths: ProjectPaths | None = None) -> Path:
    """Point HuggingFace's caches at the project cache directory.

    Setting these before importing/using ``transformers`` keeps downloaded
    artefacts on the (Drive-backed) project cache instead of the ephemeral
    home directory. Returns the cache path that was configured.
    """
    paths = paths or get_paths()
    cache = paths.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    return cache


__all__ = [
    "ProjectPaths",
    "REPO_ROOT",
    "ENV_HOME",
    "get_paths",
    "configure_hf_cache",
    "sanitize_model_id",
]
