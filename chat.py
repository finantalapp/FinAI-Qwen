#!/usr/bin/env python
"""Gradio chat interface for FinAI-Qwen.

Launches a streaming chat UI backed by :class:`src.inference.InferenceEngine`.
Supports conversation history, a configurable system prompt, and live control of
temperature, top-p, top-k, repetition penalty and max new tokens.

Examples
--------
Run with the configured (downloaded) base model::

    python chat.py

Run against a merged fine-tuned model and create a public share link::

    python chat.py --model-path "$FINAI_HOME/merged/finai-qlora-merged" --share
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# --- import shim: make ``config`` / ``src`` importable from anywhere ---------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from config.generation_config import DEFAULT_SYSTEM_PROMPT, GenerationSettings
from config.model_config import ModelConfig
from src.inference.engine import InferenceEngine
from src.utils.bootstrap import setup
from src.utils.logging_utils import get_logger

logger = get_logger("finai.chat")


def build_demo(engine: InferenceEngine, defaults: GenerationSettings):
    """Construct (but do not launch) the Gradio ChatInterface."""
    import gradio as gr

    def respond(
        message: str,
        history: list[dict[str, str]],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ):
        """Stream the assistant's reply, yielding the growing text each step."""
        settings = defaults.with_overrides(
            system_prompt=system_prompt or None,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
        )
        conversation = list(history) + [{"role": "user", "content": message}]
        accumulated = ""
        for delta in engine.stream(conversation, settings):
            accumulated += delta
            yield accumulated

    additional_inputs = [
        gr.Textbox(value=defaults.system_prompt, label="System prompt", lines=3),
        gr.Slider(16, 4096, value=defaults.max_new_tokens, step=16, label="Max new tokens"),
        gr.Slider(0.0, 2.0, value=defaults.temperature, step=0.05, label="Temperature"),
        gr.Slider(0.0, 1.0, value=defaults.top_p, step=0.01, label="Top-p"),
        gr.Slider(0, 200, value=defaults.top_k, step=1, label="Top-k"),
        gr.Slider(1.0, 2.0, value=defaults.repetition_penalty, step=0.01, label="Repetition penalty"),
    ]

    return gr.ChatInterface(
        fn=respond,
        type="messages",
        title="FinAI-Qwen",
        description=(
            "Streaming chat over the FinAI-Qwen model. Adjust the sampling "
            "controls in the accordion below to change generation behaviour."
        ),
        additional_inputs=additional_inputs,
        additional_inputs_accordion=gr.Accordion("Generation settings", open=False),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the FinAI-Qwen chat UI.")
    parser.add_argument("--model-path", default=None, help="Path to a specific (e.g. merged) model directory.")
    parser.add_argument("--model-id", default=None, help="Override the HuggingFace model id / config default.")
    parser.add_argument("--no-quantize", action="store_true", help="Load in full precision instead of 4-bit.")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup()

    model_cfg = ModelConfig()
    if args.model_id:
        model_cfg.model_id = args.model_id

    if args.model_path:
        logger.info("Loading model from explicit path: %s", args.model_path)
        engine = InferenceEngine.from_path(args.model_path, model_cfg)
    else:
        engine = InferenceEngine.from_config(
            model_cfg, quantize=False if args.no_quantize else None
        )

    demo = build_demo(engine, GenerationSettings())
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
