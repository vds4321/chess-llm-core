"""
chess-llm-core -- Shared LLM abstraction layer for chess coaching applications.

This package provides a provider-agnostic interface for working with large
language models in the context of chess coaching.  It is the shared
foundation used by `KasparChess <https://kasparchess.com>`_ and
`YourChessDotComCoach <https://yourchessdotcomcoach.fly.dev>`_.

Key capabilities:
    - **Provider abstraction** -- protocol-based interface supporting
      Anthropic, OpenAI (stub), and local models (stub).
    - **Model tier system** -- CHEAP / STANDARD / PREMIUM tiers for
      automatic cost-quality routing.
    - **Versioned prompt templates** -- reusable, testable prompts for
      coaching, scouting, and data extraction.
    - **Cost & usage tracking** -- per-request and aggregate cost
      monitoring with budget limits.

Quick start::

    from chess_llm import get_provider, ModelTier
    from chess_llm.prompts import MentorInsightsPrompt

    provider = get_provider("anthropic", tier=ModelTier.STANDARD)
    prompt = MentorInsightsPrompt(username="player", stats=stats)
    response = provider.complete(prompt.render())

Security:
    API keys are loaded from environment variables and never exposed in
    ``__repr__`` output or log messages.  See ``SECURITY.md`` for the
    full security policy.
"""

from chess_llm.providers.base import (
    LLMProvider,
    BaseLLMProvider,
    LLMResponse,
    LLMConfig,
    UsageStats,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    TokenLimitError,
)
from chess_llm.config.models import ModelTier, ModelConfig, get_model_config
from chess_llm.config.settings import Settings, get_settings
from chess_llm.providers.registry import get_provider, register_provider, list_providers
from chess_llm.tracking.usage import UsageTracker, get_tracker, reset_tracker

__version__ = "0.2.0"

__all__ = [
    # Core types
    "LLMProvider",
    "BaseLLMProvider",
    "LLMResponse",
    "LLMConfig",
    "UsageStats",
    # Errors
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "TokenLimitError",
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
