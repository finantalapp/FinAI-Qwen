"""FinAI-Qwen core library.

The :mod:`src` package contains all reusable logic, organised into focused
sub-packages:

* :mod:`src.utils` - environment detection, logging, system metrics.
* :mod:`src.data` - dataset format conversion and loading.
* :mod:`src.models` - downloading, loading and inspecting models.
* :mod:`src.inference` - the generation engine used by the chat UI and tools.
* :mod:`src.training` - the QLoRA training and LoRA-merge pipelines.
* :mod:`src.evaluation` - perplexity and qualitative evaluation.

Command-line entry points live in ``scripts/`` and import from here so that the
business logic stays testable and free of argument-parsing concerns.
"""

from __future__ import annotations

__version__ = "0.1.0"
