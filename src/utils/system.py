"""System and accelerator resource metrics.

Used by the benchmark script and surfaced in logs. All functions are safe to
call without a GPU (and without ``psutil``): missing capabilities yield ``None``
or zero rather than raising.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

_GIB = 1024**3


def human_bytes(num_bytes: float) -> str:
    """Render a byte count as a human-readable string (e.g. ``"14.2 GiB"``)."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"  # pragma: no cover - unreachable, keeps mypy happy


def _torch():
    try:
        import torch

        return torch
    except Exception:  # pragma: no cover
        return None


def reset_peak_memory() -> None:
    """Reset CUDA peak-memory counters so a benchmark measures a clean window."""
    torch = _torch()
    if torch and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


@dataclass
class VRAMStats:
    """GPU memory figures in GiB (``None`` fields mean "no CUDA device")."""

    device_name: str | None = None
    allocated_gib: float | None = None
    reserved_gib: float | None = None
    peak_allocated_gib: float | None = None
    total_gib: float | None = None


@dataclass
class RAMStats:
    """Host memory figures in GiB."""

    process_rss_gib: float | None = None
    used_gib: float | None = None
    total_gib: float | None = None
    percent: float | None = None


def vram_usage() -> VRAMStats:
    """Return current GPU memory statistics."""
    torch = _torch()
    if not (torch and torch.cuda.is_available()):
        return VRAMStats()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    return VRAMStats(
        device_name=props.name,
        allocated_gib=torch.cuda.memory_allocated(0) / _GIB,
        reserved_gib=torch.cuda.memory_reserved(0) / _GIB,
        peak_allocated_gib=torch.cuda.max_memory_allocated(0) / _GIB,
        total_gib=props.total_memory / _GIB,
    )


def ram_usage() -> RAMStats:
    """Return host RAM statistics, using ``psutil`` when available."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        return RAMStats(
            process_rss_gib=proc.memory_info().rss / _GIB,
            used_gib=vm.used / _GIB,
            total_gib=vm.total / _GIB,
            percent=vm.percent,
        )
    except Exception:
        # Fallback: report nothing rather than failing the caller.
        return RAMStats()


def system_report() -> dict[str, object]:
    """Combined VRAM + RAM snapshot as a plain dictionary."""
    return {"vram": asdict(vram_usage()), "ram": asdict(ram_usage())}


def format_system_report() -> str:
    """Pretty multi-line string describing current resource usage."""
    vram = vram_usage()
    ram = ram_usage()
    lines = ["System resources:"]
    if vram.device_name:
        lines += [
            f"  GPU            : {vram.device_name}",
            f"  VRAM allocated : {vram.allocated_gib:.2f} GiB",
            f"  VRAM reserved  : {vram.reserved_gib:.2f} GiB",
            f"  VRAM peak      : {vram.peak_allocated_gib:.2f} GiB",
            f"  VRAM total     : {vram.total_gib:.2f} GiB",
        ]
    else:
        lines.append("  GPU            : (none detected)")
    if ram.total_gib is not None:
        lines += [
            f"  RAM process    : {ram.process_rss_gib:.2f} GiB",
            f"  RAM used/total : {ram.used_gib:.2f} / {ram.total_gib:.2f} GiB ({ram.percent:.0f}%)",
        ]
    return "\n".join(lines)


__all__ = [
    "human_bytes",
    "reset_peak_memory",
    "VRAMStats",
    "RAMStats",
    "vram_usage",
    "ram_usage",
    "system_report",
    "format_system_report",
]
