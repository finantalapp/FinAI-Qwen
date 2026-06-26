"""Inspect a downloaded model and produce a human-readable report.

Crucially, parameter counting reads the *safetensors headers only* (an 8-byte
length prefix followed by a JSON blob) rather than loading any weights into
memory. That means :func:`build_report` works on a CPU-only box, needs no torch
or transformers, and is effectively instantaneous even for multi-billion
parameter models.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.system import human_bytes

# Number of bytes each safetensors dtype occupies per element.
_DTYPE_BYTES: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

_REPORT_FILE_SUFFIXES = (".safetensors", ".json", ".txt", ".model")


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Read and return the JSON header of a safetensors file.

    The safetensors layout is: ``<u64 little-endian header length><JSON header>``.
    The ``__metadata__`` key, if present, is stripped from the result.
    """
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"File too small to be safetensors: {path}")
        (header_len,) = struct.unpack("<Q", length_bytes)
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Truncated safetensors header in {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


@dataclass
class ModelReport:
    """Structured summary of a model directory."""

    model_dir: Path
    files: list[tuple[str, int]] = field(default_factory=list)
    total_disk_bytes: int = 0
    num_parameters: int = 0
    dtype_breakdown: dict[str, int] = field(default_factory=dict)
    architecture: str | None = None
    model_type: str | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    tokenizer_class: str | None = None
    generation_config: dict | None = None
    integrity_ok: bool = True
    issues: list[str] = field(default_factory=list)

    def format(self) -> str:
        """Render the report as an aligned, multi-line string."""
        lines = [
            "=" * 64,
            f"Model report: {self.model_dir}",
            "=" * 64,
            f"Architecture       : {self.architecture or 'unknown'} ({self.model_type or '?'})",
            f"Parameters         : {self.num_parameters:,} ({self.num_parameters / 1e9:.2f} B)",
            f"Hidden size        : {self.hidden_size}",
            f"Layers             : {self.num_layers}",
            f"Vocab size         : {self.vocab_size}",
            f"Tokenizer class    : {self.tokenizer_class or 'unknown'}",
            f"On-disk size       : {human_bytes(self.total_disk_bytes)}",
            "",
            "Parameter dtypes:",
        ]
        if self.dtype_breakdown:
            for dtype, count in sorted(self.dtype_breakdown.items(), key=lambda kv: -kv[1]):
                share = 100.0 * count / max(self.num_parameters, 1)
                lines.append(f"  {dtype:<8} : {count:>15,} ({share:5.1f}%)")
        else:
            lines.append("  (no safetensors tensors found)")

        lines += ["", "Files:"]
        for name, size in self.files:
            lines.append(f"  {name:<40} {human_bytes(size):>12}")

        if self.generation_config:
            lines += ["", "Generation config:"]
            for key in ("max_new_tokens", "temperature", "top_p", "top_k",
                        "repetition_penalty", "do_sample", "eos_token_id", "pad_token_id"):
                if key in self.generation_config:
                    lines.append(f"  {key:<20}: {self.generation_config[key]}")

        lines += ["", f"Integrity          : {'OK' if self.integrity_ok else 'PROBLEMS FOUND'}"]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        lines.append("=" * 64)
        return "\n".join(lines)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_report(model_dir: Path | str) -> ModelReport:
    """Inspect ``model_dir`` and return a :class:`ModelReport`."""
    model_dir = Path(model_dir)
    report = ModelReport(model_dir=model_dir)

    if not model_dir.is_dir():
        report.integrity_ok = False
        report.issues.append(f"Directory does not exist: {model_dir}")
        return report

    # --- files & disk size -------------------------------------------------
    for file in sorted(model_dir.rglob("*")):
        if file.is_file() and file.suffix.lower() in _REPORT_FILE_SUFFIXES:
            size = file.stat().st_size
            report.files.append((str(file.relative_to(model_dir)), size))
            report.total_disk_bytes += size

    # --- parameters from safetensors headers -------------------------------
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        report.integrity_ok = False
        report.issues.append("No .safetensors files found.")
    for shard in safetensor_files:
        try:
            header = read_safetensors_header(shard)
        except (ValueError, json.JSONDecodeError) as exc:
            report.integrity_ok = False
            report.issues.append(f"Corrupt safetensors header in {shard.name}: {exc}")
            continue
        for tensor in header.values():
            shape = tensor.get("shape", [])
            dtype = tensor.get("dtype", "?")
            numel = math.prod(shape) if shape else 1
            report.num_parameters += numel
            report.dtype_breakdown[dtype] = report.dtype_breakdown.get(dtype, 0) + numel

    # --- config.json -------------------------------------------------------
    config = _read_json(model_dir / "config.json")
    if config:
        architectures = config.get("architectures") or []
        report.architecture = architectures[0] if architectures else None
        report.model_type = config.get("model_type")
        report.hidden_size = config.get("hidden_size")
        report.num_layers = config.get("num_hidden_layers")
        report.vocab_size = config.get("vocab_size")
    else:
        report.integrity_ok = False
        report.issues.append("Missing or invalid config.json.")

    # --- tokenizer + generation config (file-based, no transformers) -------
    tok_cfg = _read_json(model_dir / "tokenizer_config.json")
    report.tokenizer_class = tok_cfg.get("tokenizer_class")
    gen_cfg = _read_json(model_dir / "generation_config.json")
    report.generation_config = gen_cfg or None

    # --- shard-index completeness -----------------------------------------
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = _read_json(index_path).get("weight_map", {})
        missing = sorted({s for s in weight_map.values() if not (model_dir / s).exists()})
        if missing:
            report.integrity_ok = False
            report.issues.append(f"Missing shards referenced by index: {', '.join(missing)}")

    return report


__all__ = ["ModelReport", "build_report", "read_safetensors_header"]
