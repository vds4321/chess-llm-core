"""
Environment-based settings for chess-llm-core.

Settings can be configured via:
1. Environment variables (CHESS_LLM_*)
2. Programmatic configuration
3. Default values

Example environment variables:
    CHESS_LLM_DEFAULT_PROVIDER=anthropic
    CHESS_LLM_DEFAULT_TIER=standard
    ANTHROPIC_API_KEY=sk-...
    OPENAI_API_KEY=sk-...
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from functools import lru_cache

from chess_llm.config.models import ModelTier


@dataclass
class ProviderSettings:
    """Settings for a specific provider."""

    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    default_model: Optional[str] = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class Settings:
    """
    Global settings for chess-llm-core.

    These settings control default behaviors across the library.
    Individual function calls can override these settings.
    """

    # Default provider and tier
    default_provider: str = "anthropic"
    default_tier: ModelTier = ModelTier.STANDARD

    # Provider-specific settings
    anthropic: ProviderSettings = field(default_factory=ProviderSettings)
    openai: ProviderSettings = field(default_factory=ProviderSettings)
    local: ProviderSettings = field(default_factory=ProviderSettings)

    # Cost tracking
    track_costs: bool = True
    cost_warning_threshold_usd: float = 1.0  # Warn if single request costs more

    # Logging
    log_requests: bool = False
    log_responses: bool = False

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000

    @classmethod
    def from_environment(cls) -> "Settings":
        """
        Create settings from environment variables.

        Environment variable mapping:
            CHESS_LLM_DEFAULT_PROVIDER -> default_provider
            CHESS_LLM_DEFAULT_TIER -> default_tier
            CHESS_LLM_TRACK_COSTS -> track_costs
            CHESS_LLM_LOG_REQUESTS -> log_requests
            ANTHROPIC_API_KEY -> anthropic.api_key
            OPENAI_API_KEY -> openai.api_key
        """
        settings = cls()

        # Global settings
        if provider := os.getenv("CHESS_LLM_DEFAULT_PROVIDER"):
            settings.default_provider = provider.lower()

        if tier := os.getenv("CHESS_LLM_DEFAULT_TIER"):
            try:
                settings.default_tier = ModelTier(tier.lower())
            except ValueError:
                pass  # Keep default

        if track := os.getenv("CHESS_LLM_TRACK_COSTS"):
            settings.track_costs = track.lower() in ("true", "1", "yes")

        if log_req := os.getenv("CHESS_LLM_LOG_REQUESTS"):
            settings.log_requests = log_req.lower() in ("true", "1", "yes")

        if log_resp := os.getenv("CHESS_LLM_LOG_RESPONSES"):
            settings.log_responses = log_resp.lower() in ("true", "1", "yes")

        # Anthropic settings
        settings.anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_base := os.getenv("ANTHROPIC_API_BASE"):
            settings.anthropic.api_base_url = anthropic_base
        if anthropic_model := os.getenv("CHESS_LLM_ANTHROPIC_DEFAULT_MODEL"):
            settings.anthropic.default_model = anthropic_model

        # OpenAI settings
        settings.openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai_base := os.getenv("OPENAI_API_BASE"):
            settings.openai.api_base_url = openai_base
        if openai_model := os.getenv("CHESS_LLM_OPENAI_DEFAULT_MODEL"):
            settings.openai.default_model = openai_model

        # Local model settings
        if local_base := os.getenv("OLLAMA_HOST"):
            settings.local.api_base_url = local_base
        if local_model := os.getenv("CHESS_LLM_LOCAL_DEFAULT_MODEL"):
            settings.local.default_model = local_model

        return settings

    def get_provider_settings(self, provider: str) -> ProviderSettings:
        """Get settings for a specific provider."""
        provider = provider.lower()
        if provider == "anthropic":
            return self.anthropic
        elif provider == "openai":
            return self.openai
        elif provider == "local":
            return self.local
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.get_provider_settings(provider).api_key


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Creates settings from environment variables on first call.
    Use configure_settings() to override programmatically.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_environment()
    return _settings


def configure_settings(settings: Settings) -> None:
    """
    Set the global settings instance.

    Args:
        settings: Settings instance to use globally
    """
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset settings to be reloaded from environment."""
    global _settings
    _settings = None
