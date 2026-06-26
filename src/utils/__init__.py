"""Cross-cutting utilities: environment, logging, system metrics, bootstrap."""

from __future__ import annotations

from src.utils.bootstrap import ensure_project_on_path, setup
from src.utils.env import (
    device,
    flash_attention_available,
    has_cuda,
    is_colab,
    is_drive_mounted,
    mount_drive,
    set_seed,
    supports_bf16,
)
from src.utils.logging_utils import get_logger
from src.utils.system import (
    format_system_report,
    human_bytes,
    ram_usage,
    reset_peak_memory,
    system_report,
    vram_usage,
)

__all__ = [
    "ensure_project_on_path",
    "setup",
    "get_logger",
    "is_colab",
    "is_drive_mounted",
    "mount_drive",
    "has_cuda",
    "device",
    "supports_bf16",
    "flash_attention_available",
    "set_seed",
    "human_bytes",
    "reset_peak_memory",
    "vram_usage",
    "ram_usage",
    "system_report",
    "format_system_report",
]
