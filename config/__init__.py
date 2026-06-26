"""FinAI-Qwen configuration package.

Importing :mod:`config` gives convenient access to every configuration object
and the project path layout::

    from config import get_paths, ModelConfig, GenerationSettings, TrainingConfig

Keeping configuration in a dedicated, dependency-light package means the rest of
the codebase never hard-codes paths or hyper-parameters.
"""

from __future__ import annotations

from config.generation_config import DEFAULT_SYSTEM_PROMPT, GenerationSettings
from config.model_config import (
    DEFAULT_ALLOW_PATTERNS,
    DEFAULT_MODEL_ID,
    ModelConfig,
)
from config.paths import (
    ProjectPaths,
    configure_hf_cache,
    get_paths,
    sanitize_model_id,
)
from config.training_config import LoRAConfig, TrainingConfig

__all__ = [
    "ProjectPaths",
    "get_paths",
    "configure_hf_cache",
    "sanitize_model_id",
    "ModelConfig",
    "DEFAULT_MODEL_ID",
    "DEFAULT_ALLOW_PATTERNS",
    "GenerationSettings",
    "DEFAULT_SYSTEM_PROMPT",
    "TrainingConfig",
    "LoRAConfig",
]
