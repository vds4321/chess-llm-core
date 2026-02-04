"""Tests for configuration modules."""

import os
import pytest
from unittest.mock import patch

from chess_llm.config.models import (
    ModelTier,
    ModelConfig,
    get_model_config,
    get_default_model,
    get_models_by_tier,
    ANTHROPIC_MODELS,
)
from chess_llm.config.settings import Settings, get_settings


class TestModelTier:
    """Test ModelTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert ModelTier.CHEAP.value == "cheap"
        assert ModelTier.STANDARD.value == "standard"
        assert ModelTier.PREMIUM.value == "premium"

    def test_tier_ordering(self):
        """Test tier comparison via value."""
        tiers = [ModelTier.PREMIUM, ModelTier.CHEAP, ModelTier.STANDARD]
        sorted_tiers = sorted(tiers, key=lambda t: t.value)
        assert sorted_tiers[0] == ModelTier.CHEAP


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_create_model_config(self):
        """Test creating a model config."""
        config = ModelConfig(
            model_id="test-model",
            provider="test",
            display_name="Test Model",
            tier=ModelTier.STANDARD,
            max_context_tokens=100000,
            max_output_tokens=4096,
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        )
        assert config.model_id == "test-model"
        assert config.tier == ModelTier.STANDARD
        assert config.input_cost_per_million == 1.0

    def test_anthropic_models_exist(self):
        """Test that Anthropic models are configured."""
        assert len(ANTHROPIC_MODELS) > 0
        assert "claude-3-5-haiku-20241022" in ANTHROPIC_MODELS


class TestGetModelConfig:
    """Test get_model_config function."""

    def test_get_known_model(self):
        """Test getting config for a known model."""
        config = get_model_config("claude-3-5-haiku-20241022")
        assert config is not None
        assert config.tier == ModelTier.CHEAP

    def test_get_unknown_model(self):
        """Test getting config for unknown model returns None."""
        config = get_model_config("nonexistent-model")
        assert config is None


class TestGetDefaultModel:
    """Test get_default_model function."""

    def test_get_cheap_model(self):
        """Test getting default cheap model."""
        model_id = get_default_model("anthropic", ModelTier.CHEAP)
        assert model_id is not None
        config = get_model_config(model_id)
        assert config.tier == ModelTier.CHEAP

    def test_get_standard_model(self):
        """Test getting default standard model."""
        model_id = get_default_model("anthropic", ModelTier.STANDARD)
        assert model_id is not None
        config = get_model_config(model_id)
        assert config.tier == ModelTier.STANDARD

    def test_get_premium_model(self):
        """Test getting default premium model."""
        model_id = get_default_model("anthropic", ModelTier.PREMIUM)
        assert model_id is not None
        config = get_model_config(model_id)
        assert config.tier == ModelTier.PREMIUM

    def test_unknown_provider_returns_none(self):
        """Test unknown provider returns None."""
        model_id = get_default_model("unknown_provider", ModelTier.STANDARD)
        assert model_id is None


class TestGetModelsByTier:
    """Test get_models_by_tier function."""

    def test_get_cheap_models(self):
        """Test getting all cheap models."""
        models = get_models_by_tier(ModelTier.CHEAP)
        assert len(models) > 0
        for model_id, config in models:
            assert config.tier == ModelTier.CHEAP


class TestSettings:
    """Test Settings class."""

    def test_settings_defaults(self):
        """Test default settings."""
        # Create settings without patching to test defaults
        settings = Settings()
        assert settings.default_provider == "anthropic"
        assert settings.default_tier == ModelTier.STANDARD

    def test_settings_has_provider_settings(self):
        """Test settings includes provider settings."""
        settings = Settings()
        assert hasattr(settings, "anthropic")
        assert hasattr(settings, "openai")


class TestGetSettings:
    """Test get_settings singleton."""

    def test_get_settings_returns_instance(self):
        """Test get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_cached(self):
        """Test get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
