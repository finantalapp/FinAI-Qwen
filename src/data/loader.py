"""Dataset loading: from raw files to a tokenised-ready ``datasets.Dataset``.

Pipeline:

1. Resolve the dataset path (absolute, repo-relative, or under the project
   ``datasets`` directory).
2. Read raw records from one or many ``.jsonl`` / ``.json`` files.
3. Normalise every record to the unified shape (:mod:`src.data.formats`).
4. Render each record to a single ``text`` string using the model's chat
   template (for chat records) or verbatim (for plain-text records).
5. Drop empty/invalid examples and split into train/eval.

The result is a :class:`DatasetBundle` that the trainer consumes directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

from config.paths import ProjectPaths, get_paths
from src.data.formats import (
    FormatName,
    UnifiedRecord,
    normalize_record,
    validate_conversation,
)
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

logger = get_logger("finai.data")

_SUPPORTED_SUFFIXES = (".jsonl", ".json")


@dataclass
class DatasetBundle:
    """A train split and an optional eval split, plus simple statistics."""

    train: "Dataset"
    eval: "Dataset | None"
    num_train: int
    num_eval: int
    text_field: str = "text"


def resolve_dataset_path(path_str: str, paths: ProjectPaths | None = None) -> Path:
    """Resolve a dataset reference to a concrete path.

    Tries, in order: the literal path, the path relative to the repository
    root, and the path relative to the project ``datasets`` directory.
    """
    paths = paths or get_paths()
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return candidate
    for base in (paths.repo_root, paths.datasets_root):
        alt = base / path_str
        if alt.exists():
            return alt
    # Return the most useful guess so the error message points somewhere real.
    return candidate


def _iter_files(path: Path) -> list[Path]:
    """Expand a file or directory into a sorted list of dataset files."""
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No .jsonl/.json files found under '{path}'.")
        return files
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: '{path}'.")
    return [path]


def iter_raw_records(path: Path) -> Iterator[dict[str, object]]:
    """Yield raw record dictionaries from a file or directory.

    ``.jsonl`` files are read line-by-line; ``.json`` files may contain either a
    list of records or a single record object.
    """
    for file in _iter_files(path):
        if file.suffix.lower() == ".jsonl":
            with file.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON on line {line_no} of '{file}': {exc}"
                        ) from exc
                    if isinstance(record, dict):
                        yield record
        else:  # .json
            with file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                yield from (item for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict):
                yield payload


def render_record(
    record: UnifiedRecord,
    tokenizer: "PreTrainedTokenizerBase",
) -> str | None:
    """Render a unified record to a single training string.

    Chat records are formatted with the tokenizer's chat template; plain-text
    records are returned verbatim. Returns ``None`` for records that are empty
    or otherwise unusable for supervised fine-tuning.
    """
    if "messages" in record:
        conversation = record["messages"]  # type: ignore[assignment]
        if not isinstance(conversation, list) or not validate_conversation(conversation):
            return None
        if tokenizer.chat_template is None:
            raise ValueError(
                "The tokenizer has no chat template, so chat-format data cannot "
                "be rendered. Use an instruct model (which ships a template) or "
                "provide plain-text data with a {'text': ...} schema."
            )
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        return text if text.strip() else None

    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return None


def prepare_dataset(
    dataset_path: str,
    tokenizer: "PreTrainedTokenizerBase",
    *,
    fmt: FormatName | Literal["auto"] = "auto",
    eval_split_ratio: float = 0.05,
    seed: int = 42,
    default_system: str | None = None,
    paths: ProjectPaths | None = None,
) -> DatasetBundle:
    """Load, normalise, render and split a dataset for SFT.

    Parameters
    ----------
    dataset_path:
        File or directory reference (resolved by :func:`resolve_dataset_path`).
    tokenizer:
        Tokenizer used to apply the chat template.
    fmt:
        Input format, or ``"auto"`` to detect per-record.
    eval_split_ratio:
        Fraction held out for evaluation (``0`` disables the eval split).
    """
    from datasets import Dataset

    resolved = resolve_dataset_path(dataset_path, paths)
    logger.info("Loading dataset from %s", resolved)

    texts: list[str] = []
    seen, kept = 0, 0
    for raw in iter_raw_records(resolved):
        seen += 1
        unified = normalize_record(raw, fmt, default_system=default_system)
        rendered = render_record(unified, tokenizer)
        if rendered is not None:
            texts.append(rendered)
            kept += 1

    if not texts:
        raise ValueError(
            f"No usable examples were produced from '{resolved}'. "
            f"Inspected {seen} records; check the dataset format."
        )
    logger.info("Prepared %d usable examples (from %d raw records).", kept, seen)

    dataset = Dataset.from_dict({"text": texts})

    # Hold out an eval split when requested and there is enough data for it.
    if eval_split_ratio > 0 and len(dataset) >= 10:
        split = dataset.train_test_split(test_size=eval_split_ratio, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]
        return DatasetBundle(train_ds, eval_ds, len(train_ds), len(eval_ds))

    return DatasetBundle(dataset, None, len(dataset), 0)


__all__ = [
    "DatasetBundle",
    "resolve_dataset_path",
    "iter_raw_records",
    "render_record",
    "prepare_dataset",
]
