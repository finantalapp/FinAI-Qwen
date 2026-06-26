"""Tests for dataset loading helpers (no torch/datasets required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.paths import get_paths
from src.data.loader import iter_raw_records, render_record, resolve_dataset_path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "data" / "samples"


class FakeTokenizer:
    """Minimal tokenizer stand-in implementing ``apply_chat_template``."""

    def __init__(self, chat_template: str | None = "fake-template") -> None:
        self.chat_template = chat_template

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in conversation)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("alpaca_sample.jsonl", 9),
        ("sharegpt_sample.jsonl", 6),
        ("chatml_sample.jsonl", 5),
        ("openai_sample.jsonl", 6),
    ],
)
def test_iter_raw_records_counts(filename: str, expected: int) -> None:
    records = list(iter_raw_records(SAMPLES / filename))
    assert len(records) == expected
    assert all(isinstance(r, dict) for r in records)


def test_iter_raw_records_directory() -> None:
    records = list(iter_raw_records(SAMPLES))
    # Sum of all four sample files.
    assert len(records) == 9 + 6 + 5 + 6


def test_resolve_dataset_path_repo_relative() -> None:
    resolved = resolve_dataset_path("data/samples/alpaca_sample.jsonl")
    assert resolved.exists()
    assert resolved.name == "alpaca_sample.jsonl"


def test_resolve_dataset_path_absolute(tmp_path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    assert resolve_dataset_path(str(f)) == f


class TestRenderRecord:
    def test_render_messages(self) -> None:
        tok = FakeTokenizer()
        record = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]}
        rendered = render_record(record, tok)
        assert rendered == "user: q\nassistant: a"

    def test_render_plain_text(self) -> None:
        assert render_record({"text": "hello"}, FakeTokenizer()) == "hello"

    def test_render_invalid_conversation_returns_none(self) -> None:
        record = {"messages": [{"role": "assistant", "content": ""}]}
        assert render_record(record, FakeTokenizer()) is None

    def test_render_empty_text_returns_none(self) -> None:
        assert render_record({"text": "   "}, FakeTokenizer()) is None

    def test_render_chat_without_template_raises(self) -> None:
        tok = FakeTokenizer(chat_template=None)
        record = {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]}
        with pytest.raises(ValueError):
            render_record(record, tok)


def test_end_to_end_normalise_and_render_samples() -> None:
    """Every sample record should normalise and render to non-empty text."""
    from src.data.formats import normalize_record

    tok = FakeTokenizer()
    for filename in ("alpaca_sample.jsonl", "sharegpt_sample.jsonl", "chatml_sample.jsonl", "openai_sample.jsonl"):
        for raw in iter_raw_records(SAMPLES / filename):
            rendered = render_record(normalize_record(raw), tok)
            assert rendered and rendered.strip()
