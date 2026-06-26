"""Runtime environment detection.

Every function degrades gracefully when optional dependencies (torch, the Colab
runtime, flash-attention) are missing, returning conservative defaults so the
module can be imported anywhere - including on a plain CPU box or in CI.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

_DRIVE_MOUNT = "/content/drive"


def is_colab() -> bool:
    """Return ``True`` when running inside Google Colab."""
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def is_drive_mounted() -> bool:
    """Return ``True`` when Google Drive is mounted under ``/content/drive``."""
    return os.path.isdir(os.path.join(_DRIVE_MOUNT, "MyDrive"))


def mount_drive(mount_point: str = _DRIVE_MOUNT, *, force_remount: bool = False) -> bool:
    """Mount Google Drive when on Colab; no-op elsewhere.

    Returns ``True`` if Drive is mounted afterwards. Outside Colab this returns
    ``False`` without raising, so callers can use it unconditionally.
    """
    if not is_colab():
        return False
    if is_drive_mounted() and not force_remount:
        return True
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point, force_remount=force_remount)
    return is_drive_mounted()


def _torch():
    """Import torch lazily, returning ``None`` if it is unavailable."""
    try:
        import torch

        return torch
    except Exception:  # pragma: no cover - torch genuinely missing
        return None


def has_cuda() -> bool:
    """Return ``True`` when a CUDA device is available."""
    torch = _torch()
    return bool(torch and torch.cuda.is_available())


def device() -> str:
    """Return the best available device string (``"cuda"`` or ``"cpu"``)."""
    return "cuda" if has_cuda() else "cpu"


@lru_cache(maxsize=1)
def cuda_capability() -> tuple[int, int] | None:
    """Return the compute capability of GPU 0, or ``None`` without CUDA."""
    torch = _torch()
    if not (torch and torch.cuda.is_available()):
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


def gpu_name() -> str | None:
    """Return the name of GPU 0, or ``None`` when no GPU is present."""
    torch = _torch()
    if not (torch and torch.cuda.is_available()):
        return None
    return torch.cuda.get_device_name(0)


def supports_bf16() -> bool:
    """Return ``True`` when the current GPU supports bfloat16 training."""
    torch = _torch()
    if not (torch and torch.cuda.is_available()):
        return False
    try:
        if torch.cuda.is_bf16_supported():
            return True
    except Exception:  # pragma: no cover - older torch without the helper
        pass
    cap = cuda_capability()
    # Ampere (SM 8.0) and newer have native bf16 support.
    return cap is not None and cap[0] >= 8


def flash_attention_available() -> bool:
    """Return ``True`` when flash-attention-2 can actually be used.

    Requires the ``flash_attn`` package to be importable *and* an Ampere-or-newer
    GPU (compute capability >= 8.0). We only check that the module is installed
    (via :func:`importlib.util.find_spec`) rather than importing it, because the
    import is expensive and can fail loudly on unsupported hardware.
    """
    if importlib.util.find_spec("flash_attn") is None:
        return False
    cap = cuda_capability()
    return cap is not None and cap[0] >= 8


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy optional here
        pass
    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def summary() -> dict[str, object]:
    """Return a compact dictionary describing the runtime environment."""
    return {
        "colab": is_colab(),
        "drive_mounted": is_drive_mounted(),
        "cuda": has_cuda(),
        "gpu_name": gpu_name(),
        "cuda_capability": cuda_capability(),
        "bf16_supported": supports_bf16(),
        "flash_attention": flash_attention_available(),
        "python": sys.version.split()[0],
    }


__all__ = [
    "is_colab",
    "is_drive_mounted",
    "mount_drive",
    "has_cuda",
    "device",
    "cuda_capability",
    "gpu_name",
    "supports_bf16",
    "flash_attention_available",
    "set_seed",
    "summary",
]
