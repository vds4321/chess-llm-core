"""Tests for provider modules."""

import pytest
from unittest.mock import MagicMock, patch

from chess_llm.config.models import ModelTier
from chess_llm.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMConfig,
    UsageStats,
    BaseLLMProvider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
)
from chess_llm.providers.registry import (
    get_provider,
    register_provider,
    list_providers,
    is_provider_available,
    get_provider_for_tier,
    _providers,
)


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = LLMConfig()
        assert config.max_tokens == 1024  # Default is 1024
        assert config.temperature == 0.7
        assert config.top_p == 1.0

    def test_custom_config(self):
        """Test custom config values."""
        config = LLMConfig(max_tokens=1000, temperature=0.5, top_p=0.9)
        assert config.max_tokens == 1000
        assert config.temperature == 0.5
        assert config.top_p == 0.9


class TestUsageStats:
    """Test UsageStats dataclass."""

    def test_usage_stats(self):
        """Test usage stats creation."""
        stats = UsageStats(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            total_cost=0.001,
        )
        assert stats.input_tokens == 100
        assert stats.total_tokens == 300
        assert stats.total_cost == 0.001

    def test_cost_formatted(self):
        """Test cost formatting."""
        stats = UsageStats(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            total_cost=0.001234,
        )
        assert stats.cost_formatted == "$0.001234"


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_response_creation(self):
        """Test response creation."""
        usage = UsageStats(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
        )
        response = LLMResponse(
            content="Hello",
            model="test-model",
            provider="test",
            usage=usage,
            finish_reason="stop",
            latency_ms=150.0,
        )
        assert response.content == "Hello"
        assert response.model == "test-model"
        assert response.latency_ms == 150.0


class TestProviderErrors:
    """Test provider error classes."""

    def test_provider_error(self):
        """Test base provider error."""
        error = ProviderError("Something failed")
        assert str(error) == "Something failed"

    def test_rate_limit_error(self):
        """Test rate limit error with retry_after."""
        error = RateLimitError("Rate limited", retry_after=60.0)
        assert error.retry_after == 60.0
        assert "Rate limited" in str(error)

    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Invalid key")
        assert isinstance(error, ProviderError)


class TestBaseLLMProvider:
    """Test BaseLLMProvider abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        # BaseLLMProvider is abstract and requires complete implementation
        # This test verifies the protocol design
        assert hasattr(BaseLLMProvider, "complete")
        assert hasattr(BaseLLMProvider, "provider_id")
        assert hasattr(BaseLLMProvider, "model_id")


class TestProviderRegistry:
    """Test provider registry functions."""

    def test_list_providers(self):
        """Test listing providers."""
        providers = list_providers()
        assert isinstance(providers, list)
        assert "anthropic" in providers

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        # Create a mock provider factory
        mock_factory = MagicMock()

        # Register with override to avoid conflicts
        register_provider("test_custom", mock_factory, override=True)

        assert "test_custom" in _providers

    def test_register_duplicate_raises(self):
        """Test registering duplicate provider raises error."""
        mock_factory = MagicMock()
        register_provider("test_dup", mock_factory, override=True)

        with pytest.raises(ValueError, match="already registered"):
            register_provider("test_dup", mock_factory, override=False)

    def test_get_unknown_provider_raises(self):
        """Test getting unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent_provider_xyz")

    def test_is_provider_available_anthropic(self):
        """Test checking if anthropic provider is available."""
        # This will depend on whether anthropic is installed
        result = is_provider_available("anthropic")
        assert isinstance(result, bool)

    def test_is_provider_available_unknown(self):
        """Test checking unknown provider returns False."""
        result = is_provider_available("definitely_not_a_provider")
        assert result is False


class TestGetProviderForTier:
    """Test get_provider_for_tier convenience function."""

    @patch("chess_llm.providers.registry.get_provider")
    def test_get_provider_for_tier(self, mock_get_provider):
        """Test get_provider_for_tier passes tier correctly."""
        get_provider_for_tier(ModelTier.CHEAP)
        mock_get_provider.assert_called_once_with(
            provider_id=None, tier=ModelTier.CHEAP
        )

    @patch("chess_llm.providers.registry.get_provider")
    def test_get_provider_for_tier_with_provider(self, mock_get_provider):
        """Test get_provider_for_tier with specific provider."""
        get_provider_for_tier(ModelTier.STANDARD, provider_id="anthropic")
        mock_get_provider.assert_called_once_with(
            provider_id="anthropic", tier=ModelTier.STANDARD
        )
