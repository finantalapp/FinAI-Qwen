"""Tests for dataset format detection and normalisation."""

from __future__ import annotations

import pytest

from src.data.formats import (
    FormatError,
    alpaca_to_messages,
    chatml_to_messages,
    detect_format,
    normalize_record,
    openai_to_messages,
    prompt_completion_to_messages,
    sharegpt_to_messages,
    validate_conversation,
)


class TestDetectFormat:
    def test_openai(self) -> None:
        assert detect_format({"messages": [{"role": "user", "content": "hi"}]}) == "openai"

    def test_sharegpt(self) -> None:
        assert detect_format({"conversations": [{"from": "human", "value": "hi"}]}) == "sharegpt"

    def test_alpaca(self) -> None:
        assert detect_format({"instruction": "do x", "output": "y"}) == "alpaca"

    def test_prompt_completion(self) -> None:
        assert detect_format({"prompt": "p", "completion": "c"}) == "prompt"
        assert detect_format({"prompt": "p", "response": "c"}) == "prompt"

    def test_chatml(self) -> None:
        assert detect_format({"text": "<|im_start|>user\nhi<|im_end|>"}) == "chatml"

    def test_plain_text(self) -> None:
        assert detect_format({"text": "just some text"}) == "text"

    def test_unknown_raises(self) -> None:
        with pytest.raises(FormatError):
            detect_format({"foo": "bar"})


class TestConverters:
    def test_alpaca_without_input(self) -> None:
        msgs = alpaca_to_messages({"instruction": "Say hi", "output": "Hi!"})
        assert msgs == [
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_alpaca_with_input_and_system(self) -> None:
        msgs = alpaca_to_messages(
            {"instruction": "Summarise", "input": "Long text", "output": "Short", "system": "Be terse"}
        )
        assert msgs[0] == {"role": "system", "content": "Be terse"}
        assert msgs[1]["content"] == "Summarise\n\nLong text"
        assert msgs[2] == {"role": "assistant", "content": "Short"}

    def test_alpaca_default_system(self) -> None:
        msgs = alpaca_to_messages({"instruction": "x", "output": "y"}, default_system="SYS")
        assert msgs[0] == {"role": "system", "content": "SYS"}

    def test_sharegpt_role_mapping(self) -> None:
        msgs = sharegpt_to_messages(
            {"conversations": [
                {"from": "system", "value": "s"},
                {"from": "human", "value": "h"},
                {"from": "gpt", "value": "g"},
            ]}
        )
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]

    def test_sharegpt_top_level_system(self) -> None:
        msgs = sharegpt_to_messages(
            {"system": "S", "conversations": [{"from": "human", "value": "h"}, {"from": "gpt", "value": "g"}]}
        )
        assert msgs[0] == {"role": "system", "content": "S"}

    def test_openai_passthrough(self) -> None:
        record = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
        assert openai_to_messages(record) == record["messages"]

    def test_openai_content_parts(self) -> None:
        record = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}
        assert openai_to_messages(record)[0]["content"] == "a\nb"

    def test_prompt_completion_and_response(self) -> None:
        assert prompt_completion_to_messages({"prompt": "p", "completion": "c"})[1]["content"] == "c"
        assert prompt_completion_to_messages({"prompt": "p", "response": "r"})[1]["content"] == "r"

    def test_chatml_parsing(self) -> None:
        text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\nA<|im_end|>"
        msgs = chatml_to_messages(text)
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert [m["content"] for m in msgs] == ["S", "U", "A"]

    def test_chatml_no_turns_raises(self) -> None:
        with pytest.raises(FormatError):
            chatml_to_messages("no turns here")


class TestNormalizeRecord:
    def test_dispatch_alpaca(self) -> None:
        out = normalize_record({"instruction": "x", "output": "y"})
        assert "messages" in out

    def test_dispatch_plain_text(self) -> None:
        out = normalize_record({"text": "hello world"})
        assert out == {"text": "hello world"}

    def test_explicit_format(self) -> None:
        out = normalize_record({"prompt": "p", "completion": "c"}, fmt="prompt")
        assert out["messages"][0]["content"] == "p"


class TestValidateConversation:
    def test_valid(self) -> None:
        assert validate_conversation([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ])

    def test_missing_assistant_content(self) -> None:
        assert not validate_conversation([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "   "},
        ])

    def test_missing_user(self) -> None:
        assert not validate_conversation([{"role": "assistant", "content": "a"}])
