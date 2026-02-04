"""
Chess LLM Core - Shared LLM abstraction layer for chess coaching applications.

This package provides:
- Protocol-based LLM provider abstraction
- Multiple provider implementations (Anthropic, OpenAI, local models)
- Versioned prompt templates for chess coaching
- Model tier configuration (cheap/standard/premium)
- Cost and usage tracking

Usage:
    from chess_llm import get_provider, ModelTier
    from chess_llm.prompts import MentorInsightsPrompt

    # Get a provider for a specific tier
    provider = get_provider(ModelTier.STANDARD)

    # Use a prompt template
    prompt = MentorInsightsPrompt(username="player", stats=stats)
    response = provider.complete(prompt.render())
"""

from chess_llm.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMConfig,
    ProviderError,
    RateLimitError,
    AuthenticationError,
)
from chess_llm.config.models import ModelTier, ModelConfig, get_model_config
from chess_llm.config.settings import Settings, get_settings
from chess_llm.providers.registry import get_provider, register_provider, list_providers
from chess_llm.tracking.usage import UsageTracker, get_tracker, reset_tracker

__version__ = "0.1.0"

__all__ = [
    # Core types
    "LLMProvider",
    "LLMResponse",
    "LLMConfig",
    # Errors
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    # Configuration
    "ModelTier",
    "ModelConfig",
    "get_model_config",
    "Settings",
    "get_settings",
    # Registry
    "get_provider",
    "register_provider",
    "list_providers",
    # Tracking
    "UsageTracker",
    "get_tracker",
    "reset_tracker",
]
