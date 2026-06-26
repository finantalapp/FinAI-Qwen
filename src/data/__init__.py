"""Dataset format conversion and loading."""

from __future__ import annotations

from src.data.formats import (
    Conversation,
    FormatError,
    Message,
    UnifiedRecord,
    detect_format,
    normalize_record,
    normalize_records,
    validate_conversation,
)
from src.data.loader import (
    DatasetBundle,
    iter_raw_records,
    prepare_dataset,
    render_record,
    resolve_dataset_path,
)

__all__ = [
    "Message",
    "Conversation",
    "UnifiedRecord",
    "FormatError",
    "detect_format",
    "normalize_record",
    "normalize_records",
    "validate_conversation",
    "DatasetBundle",
    "resolve_dataset_path",
    "iter_raw_records",
    "render_record",
    "prepare_dataset",
]
