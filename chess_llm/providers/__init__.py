"""
LLM provider sub-package.

Available providers:
    - **AnthropicProvider** -- Claude models (Haiku, Sonnet 4, Opus).
      Fully implemented.
    - **OpenAI** -- GPT models.  Registered but not yet implemented.
    - **Local** -- Ollama / local models.  Registered but not yet
      implemented.

Use ``get_provider()`` from the registry rather than importing concrete
classes directly -- this keeps consuming code provider-agnostic.
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
from chess_llm.providers.registry import (
    get_provider,
    register_provider,
    list_providers,
    get_provider_for_tier,
)

__all__ = [
    "LLMProvider",
    "BaseLLMProvider",
    "LLMResponse",
    "LLMConfig",
    "UsageStats",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "TokenLimitError",
    "get_provider",
    "register_provider",
    "list_providers",
    "get_provider_for_tier",
]
