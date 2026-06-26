"""Dataset format detection and normalisation.

The project accepts data in several popular conversational/instruction formats
and normalises every record into one of two *unified* shapes:

* a chat record - ``{"messages": [{"role": ..., "content": ...}, ...]}``
* a plain-text record - ``{"text": "..."}`` (for raw continued-pretraining data)

Supported input formats (auto-detected from a record's keys):

================  ===========================================================
Format            Shape
================  ===========================================================
``openai``        ``{"messages": [{"role", "content"}, ...]}``
``sharegpt``      ``{"conversations": [{"from", "value"}, ...]}``
``alpaca``        ``{"instruction", "input"?, "output", "system"?}``
``chatml``        ``{"text": "<|im_start|>user\\n...<|im_end|>..."}``
``prompt``        ``{"prompt", "completion"|"response", "system"?}``
``text``          ``{"text": "raw text"}``
================  ===========================================================

The unified ``messages`` shape is exactly what ``tokenizer.apply_chat_template``
consumes, so downstream code never branches on the original format again.
"""

from __future__ import annotations

import re
from typing import Final, Iterable, Literal

Message = dict[str, str]
Conversation = list[Message]
UnifiedRecord = dict[str, object]

FormatName = Literal["openai", "sharegpt", "alpaca", "chatml", "prompt", "text"]

VALID_ROLES: Final[frozenset[str]] = frozenset({"system", "user", "assistant", "tool"})

# Maps the free-form ``from`` field used by ShareGPT to canonical roles.
_ROLE_ALIASES: Final[dict[str, str]] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "bard": "assistant",
    "bing": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}

# Matches a single ChatML turn, capturing the role and its content.
_CHATML_TURN = re.compile(
    r"<\|im_start\|>\s*(?P<role>[a-zA-Z]+)\s*\n(?P<content>.*?)<\|im_end\|>",
    re.DOTALL,
)


class FormatError(ValueError):
    """Raised when a record cannot be parsed into the unified shape."""


def _canonical_role(raw: str) -> str:
    return _ROLE_ALIASES.get(raw.strip().lower(), "user")


def _message(role: str, content: str) -> Message:
    return {"role": role, "content": content}


def detect_format(record: dict[str, object]) -> FormatName:
    """Infer the format of a single record from its keys.

    Raises
    ------
    FormatError
        If none of the supported shapes match.
    """
    if isinstance(record.get("messages"), list):
        return "openai"
    if isinstance(record.get("conversations"), list):
        return "sharegpt"
    if "instruction" in record:
        return "alpaca"
    if "prompt" in record and ("completion" in record or "response" in record):
        return "prompt"
    text = record.get("text")
    if isinstance(text, str):
        return "chatml" if "<|im_start|>" in text else "text"
    raise FormatError(
        f"Could not detect dataset format from record keys: {sorted(record)}"
    )


def openai_to_messages(record: dict[str, object]) -> Conversation:
    """Normalise an OpenAI-style ``{"messages": [...]}`` record."""
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        raise FormatError("OpenAI record is missing a 'messages' list.")
    conversation: Conversation = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise FormatError("Each message must be an object with role/content.")
        role = str(item.get("role", "")).strip().lower()
        content = item.get("content", "")
        if role not in VALID_ROLES:
            role = _canonical_role(role)
        conversation.append(_message(role, _stringify_content(content)))
    return conversation


def sharegpt_to_messages(record: dict[str, object]) -> Conversation:
    """Normalise a ShareGPT-style ``{"conversations": [...]}`` record."""
    turns = record.get("conversations")
    if not isinstance(turns, list):
        raise FormatError("ShareGPT record is missing a 'conversations' list.")
    conversation: Conversation = []
    # Some ShareGPT exports carry a separate top-level system prompt.
    system = record.get("system") or record.get("system_prompt")
    if isinstance(system, str) and system.strip():
        conversation.append(_message("system", system.strip()))
    for turn in turns:
        if not isinstance(turn, dict):
            raise FormatError("Each ShareGPT turn must be an object.")
        role = _canonical_role(str(turn.get("from", "user")))
        content = _stringify_content(turn.get("value", ""))
        conversation.append(_message(role, content))
    return conversation


def alpaca_to_messages(
    record: dict[str, object], *, default_system: str | None = None
) -> Conversation:
    """Normalise an Alpaca-style instruction/input/output record."""
    instruction = _stringify_content(record.get("instruction", "")).strip()
    extra_input = _stringify_content(record.get("input", "")).strip()
    output = _stringify_content(record.get("output", "")).strip()
    if not instruction:
        raise FormatError("Alpaca record has an empty 'instruction'.")

    conversation: Conversation = []
    system = record.get("system") or default_system
    if isinstance(system, str) and system.strip():
        conversation.append(_message("system", system.strip()))

    user_content = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
    conversation.append(_message("user", user_content))
    conversation.append(_message("assistant", output))
    return conversation


def prompt_completion_to_messages(record: dict[str, object]) -> Conversation:
    """Normalise a ``{"prompt", "completion"|"response"}`` record."""
    prompt = _stringify_content(record.get("prompt", "")).strip()
    completion = record.get("completion")
    if completion is None:
        completion = record.get("response", "")
    completion = _stringify_content(completion).strip()
    if not prompt:
        raise FormatError("prompt/completion record has an empty 'prompt'.")

    conversation: Conversation = []
    system = record.get("system")
    if isinstance(system, str) and system.strip():
        conversation.append(_message("system", system.strip()))
    conversation.append(_message("user", prompt))
    conversation.append(_message("assistant", completion))
    return conversation


def chatml_to_messages(text: str) -> Conversation:
    """Parse a raw ChatML string into a list of messages."""
    conversation: Conversation = [
        _message(_canonical_role(m.group("role")), m.group("content").strip())
        for m in _CHATML_TURN.finditer(text)
    ]
    if not conversation:
        raise FormatError("No '<|im_start|>...<|im_end|>' turns found in ChatML text.")
    return conversation


def _stringify_content(content: object) -> str:
    """Coerce message content to a string.

    Handles the OpenAI "content parts" form (a list of ``{"type", "text"}``
    blocks) by concatenating the text parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def normalize_record(
    record: dict[str, object],
    fmt: FormatName | Literal["auto"] = "auto",
    *,
    default_system: str | None = None,
) -> UnifiedRecord:
    """Convert any supported record into a unified record.

    Returns either ``{"messages": [...]}`` or ``{"text": "..."}``.
    """
    resolved: FormatName = detect_format(record) if fmt == "auto" else fmt

    if resolved == "openai":
        return {"messages": openai_to_messages(record)}
    if resolved == "sharegpt":
        return {"messages": sharegpt_to_messages(record)}
    if resolved == "alpaca":
        return {"messages": alpaca_to_messages(record, default_system=default_system)}
    if resolved == "prompt":
        return {"messages": prompt_completion_to_messages(record)}
    if resolved == "chatml":
        return {"messages": chatml_to_messages(str(record.get("text", "")))}
    if resolved == "text":
        return {"text": _stringify_content(record.get("text", ""))}
    raise FormatError(f"Unsupported format: {resolved!r}")


def normalize_records(
    records: Iterable[dict[str, object]],
    fmt: FormatName | Literal["auto"] = "auto",
    *,
    default_system: str | None = None,
    skip_errors: bool = True,
) -> list[UnifiedRecord]:
    """Normalise an iterable of records, optionally skipping malformed ones."""
    out: list[UnifiedRecord] = []
    for index, record in enumerate(records):
        try:
            out.append(normalize_record(record, fmt, default_system=default_system))
        except FormatError:
            if not skip_errors:
                raise
            # Silently dropping is intentional for robustness on large dumps;
            # callers that need strictness pass ``skip_errors=False``.
            continue
    return out


def validate_conversation(conversation: Conversation) -> bool:
    """Return ``True`` when a conversation is well-formed and usable for SFT.

    A usable conversation has at least one user turn and one non-empty
    assistant turn (so there is something to learn to generate).
    """
    has_user = any(m["role"] == "user" for m in conversation)
    has_assistant_content = any(
        m["role"] == "assistant" and m["content"].strip() for m in conversation
    )
    return has_user and has_assistant_content


__all__ = [
    "Message",
    "Conversation",
    "UnifiedRecord",
    "FormatName",
    "FormatError",
    "detect_format",
    "openai_to_messages",
    "sharegpt_to_messages",
    "alpaca_to_messages",
    "prompt_completion_to_messages",
    "chatml_to_messages",
    "normalize_record",
    "normalize_records",
    "validate_conversation",
]
