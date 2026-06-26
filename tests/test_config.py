"""Tests for the configuration package and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.generation_config import GenerationSettings
from config.model_config import DEFAULT_MODEL_ID, ModelConfig
from config.paths import get_paths, sanitize_model_id
from config.training_config import TrainingConfig


@pytest.fixture(autouse=True)
def _clear_paths_cache():
    """Ensure each test sees a freshly resolved ProjectPaths."""
    get_paths.cache_clear()
    yield
    get_paths.cache_clear()


class TestPaths:
    def test_sanitize_model_id(self) -> None:
        assert sanitize_model_id("Qwen/Qwen2.5-7B-Instruct") == "Qwen2.5-7B-Instruct"
        assert sanitize_model_id("local-model") == "local-model"

    def test_home_override(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FINAI_HOME", str(tmp_path))
        get_paths.cache_clear()
        paths = get_paths()
        assert paths.home == tmp_path.resolve()
        assert paths.models_root == tmp_path.resolve() / "models"
        assert paths.logs_dir == tmp_path.resolve() / "logs"

    def test_derived_dirs(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FINAI_HOME", str(tmp_path))
        get_paths.cache_clear()
        paths = get_paths()
        assert paths.model_dir("Qwen/Qwen2.5-7B-Instruct").name == "Qwen2.5-7B-Instruct"
        assert paths.adapter_dir("run1") == paths.adapters_root / "run1"
        assert paths.checkpoint_dir("run1") == paths.checkpoints_root / "run1"
        assert paths.merged_dir("run1").name == "run1-merged"

    def test_ensure_creates_dirs(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FINAI_HOME", str(tmp_path))
        get_paths.cache_clear()
        paths = get_paths().ensure()
        for directory in paths.base_dirs:
            assert directory.is_dir()


class TestModelConfig:
    def test_default_model_id(self) -> None:
        assert ModelConfig().model_id == DEFAULT_MODEL_ID

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("FINAI_MODEL_ID", "Qwen/Qwen3-8B")
        assert ModelConfig().model_id == "Qwen/Qwen3-8B"

    def test_model_name(self) -> None:
        cfg = ModelConfig(model_id="org/My-Model")
        assert cfg.model_name == "My-Model"

    def test_to_dict_roundtrip_keys(self) -> None:
        data = ModelConfig().to_dict()
        assert {"model_id", "dtype", "load_in_4bit", "max_seq_length"} <= set(data)


class TestGenerationSettings:
    def test_sampling_kwargs(self) -> None:
        kwargs = GenerationSettings(temperature=0.7).to_kwargs()
        assert kwargs["do_sample"] is True
        assert "temperature" in kwargs and "top_p" in kwargs and "top_k" in kwargs

    def test_greedy_when_zero_temperature(self) -> None:
        settings = GenerationSettings(temperature=0.0)
        assert settings.do_sample is False
        kwargs = settings.to_kwargs()
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs  # sampling knobs omitted

    def test_with_overrides_ignores_none(self) -> None:
        base = GenerationSettings(temperature=0.5)
        updated = base.with_overrides(temperature=None, top_p=0.9)
        assert updated.temperature == 0.5
        assert updated.top_p == 0.9


class TestTrainingConfig:
    def test_effective_batch_size(self) -> None:
        cfg = TrainingConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=8
        )
        assert cfg.effective_batch_size() == 16
        assert cfg.effective_batch_size(world_size=2) == 32

    def test_lora_peft_kwargs(self) -> None:
        kwargs = TrainingConfig().lora.to_peft_kwargs()
        assert kwargs["task_type"] == "CAUSAL_LM"
        assert "q_proj" in kwargs["target_modules"]

    def test_to_dict_contains_lora(self) -> None:
        data = TrainingConfig().to_dict()
        assert "lora" in data and "run_name" in data
