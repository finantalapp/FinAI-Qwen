"""The generation engine shared by the chat UI, benchmark and evaluation.

:class:`InferenceEngine` wraps a loaded ``(model, tokenizer)`` pair and exposes
two generation modes built on the model's chat template:

* :meth:`generate` - blocking, returns the full completion string.
* :meth:`stream` - yields incremental text deltas (used for live chat).

Both accept a :class:`~config.generation_config.GenerationSettings`, so sampling
behaviour is configured in exactly one place.
"""

from __future__ import annotations

from threading import Thread
from typing import TYPE_CHECKING, Iterator

from config.generation_config import GenerationSettings
from config.model_config import ModelConfig
from config.paths import ProjectPaths
from src.data.formats import Conversation, Message
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = get_logger("finai.inference")


class InferenceEngine:
    """Stateless text-generation wrapper around a model + tokenizer."""

    def __init__(self, model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase") -> None:
        self.model = model
        self.tokenizer = tokenizer

    # --- constructors ------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig | None = None,
        *,
        quantize: bool | None = None,
        paths: ProjectPaths | None = None,
    ) -> "InferenceEngine":
        """Build an engine from the project configuration (local copy first)."""
        from src.models.loader import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            model_cfg, for_training=False, quantize=quantize, padding_side="left", paths=paths
        )
        return cls(model, tokenizer)

    @classmethod
    def from_path(cls, model_path: str, model_cfg: ModelConfig | None = None) -> "InferenceEngine":
        """Build an engine from an explicit (e.g. merged) model directory."""
        from src.models.loader import load_plain_model

        model, tokenizer = load_plain_model(model_path, model_cfg)
        return cls(model, tokenizer)

    # --- helpers -----------------------------------------------------------
    @property
    def device(self) -> "torch.device":
        """Device the model's parameters live on."""
        return next(self.model.parameters()).device

    @staticmethod
    def _with_system(messages: Conversation, system_prompt: str | None) -> Conversation:
        """Prepend ``system_prompt`` unless the conversation already has one."""
        msgs: list[Message] = list(messages)
        if system_prompt and (not msgs or msgs[0].get("role") != "system"):
            msgs.insert(0, {"role": "system", "content": system_prompt})
        return msgs

    def build_inputs(self, messages: Conversation, system_prompt: str | None) -> dict:
        """Tokenise a conversation into model-ready, device-placed tensors.

        Returns a dict with ``input_ids`` and ``attention_mask``. Recent
        ``transformers`` versions return a ``BatchEncoding`` from
        ``apply_chat_template`` when ``return_dict=True``; older versions only
        return the ids tensor, which we wrap with an all-ones mask.
        """
        conversation = self._with_system(messages, system_prompt)
        try:
            encoded = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = dict(encoded)
        except TypeError:
            # Older transformers without ``return_dict`` support.
            import torch

            input_ids = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt"
            )
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        return {key: value.to(self.device) for key, value in inputs.items()}

    def _generation_kwargs(self, settings: GenerationSettings) -> dict:
        kwargs = settings.to_kwargs()
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return kwargs

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens ``text`` encodes to (no special tokens)."""
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    # --- generation --------------------------------------------------------
    def generate(self, messages: Conversation, settings: GenerationSettings | None = None) -> str:
        """Generate a full completion and return it as a string."""
        import torch

        settings = settings or GenerationSettings()
        inputs = self.build_inputs(messages, settings.system_prompt)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(**inputs, **self._generation_kwargs(settings))

        completion_ids = output[0][prompt_len:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    def stream(
        self, messages: Conversation, settings: GenerationSettings | None = None
    ) -> Iterator[str]:
        """Yield text deltas as they are generated (for live UIs).

        Generation runs in a background thread so tokens can be consumed from
        the :class:`~transformers.TextIteratorStreamer` as soon as they appear.
        """
        from transformers import TextIteratorStreamer

        settings = settings or GenerationSettings()
        inputs = self.build_inputs(messages, settings.system_prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {**inputs, **self._generation_kwargs(settings), "streamer": streamer}

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        try:
            for delta in streamer:
                if delta:
                    yield delta
        finally:
            thread.join()


__all__ = ["InferenceEngine"]
