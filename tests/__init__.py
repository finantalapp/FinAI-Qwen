"""Test suite for FinAI-Qwen.

The tests here are intentionally dependency-light: they exercise the pure-Python
logic (configuration, path resolution, dataset format conversion, record
rendering) and do not require torch, transformers or a GPU, so they run quickly
in CI and on a laptop.
"""
