"""
Environment-based settings for chess-llm-core.

This module manages all library configuration through a layered approach:

    1. Default values (defined in dataclass fields)
    2. Environment variables (loaded via ``Settings.from_environment()``)
    3. Programmatic overrides (via ``configure_settings()``)

Supported environment variables::

    CHESS_LLM_DEFAULT_PROVIDER        - LLM provider name (default: "anthropic")
    CHESS_LLM_DEFAULT_TIER            - Model tier: cheap | standard | premium
    CHESS_LLM_TRACK_COSTS             - Enable cost tracking (true/false)
    CHESS_LLM_LOG_REQUESTS            - Log outgoing prompts (true/false)
    CHESS_LLM_LOG_RESPONSES           - Log LLM responses (true/false)
    ANTHROPIC_API_KEY                 - Anthropic API key
    ANTHROPIC_API_BASE                - Custom Anthropic API base URL
    CHESS_LLM_ANTHROPIC_DEFAULT_MODEL - Override default Anthropic model
    OPENAI_API_KEY                    - OpenAI API key
    OPENAI_API_BASE                   - Custom OpenAI API base URL
    CHESS_LLM_OPENAI_DEFAULT_MODEL    - Override default OpenAI model
    OLLAMA_HOST                       - Local Ollama server URL
    CHESS_LLM_LOCAL_DEFAULT_MODEL     - Override default local model

Security:
    API keys are read exclusively from environment variables and never
    logged, serialized, or included in ``__repr__`` output.  Consuming
    applications should store keys in environment variables or a secrets
    manager -- never hard-code them in source files.

Maintenance:
    To add a new provider:
        1. Add a ``ProviderSettings`` field to ``Settings``.
        2. Wire up environment variable loading in ``from_environment()``.
        3. Add the provider name to the ``_PROVIDER_MAP`` in
           ``get_provider_settings()``.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from chess_llm.config.models import ModelTier

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_MASKED = "****"


def _mask_key(key: Optional[str]) -> str:
    """Return a masked representation of an API key for safe logging.

    Shows only the first 4 and last 4 characters (e.g. ``"sk-a...xyz9"``).
    Keys shorter than 8 characters are fully masked.
    """
    if not key:
        return "None"
    if len(key) <= 8:
        return _MASKED
    return f"{key[:4]}...{key[-4:]}"


# ---------------------------------------------------------------------------
# Provider-level settings
# ---------------------------------------------------------------------------


@dataclass
class ProviderSettings:
    """
    Connection and behaviour settings for a single LLM provider.

    Attributes:
        api_key:          API authentication key.  Loaded from environment
                          variables -- never hard-code in source.
        api_base_url:     Override the default API endpoint (useful for
                          proxies or on-premise deployments).
        default_model:    Model ID to use when none is explicitly requested.
        timeout_seconds:  Per-request HTTP timeout in seconds.
        max_retries:      Number of automatic retries on transient errors.
        extra:            Arbitrary key-value pairs forwarded to the
                          provider client constructor.
    """

    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    default_model: Optional[str] = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    extra: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Mask the API key to prevent accidental exposure in logs."""
        return (
            f"ProviderSettings(api_key={_mask_key(self.api_key)!r}, "
            f"api_base_url={self.api_base_url!r}, "
            f"default_model={self.default_model!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries})"
        )


# ---------------------------------------------------------------------------
# Global library settings
# ---------------------------------------------------------------------------

# Maps provider name -> Settings attribute name.  Used by
# ``get_provider_settings()`` to avoid repetitive if/elif chains.
_PROVIDER_MAP = {"anthropic", "openai", "local"}


@dataclass
class Settings:
    """
    Global settings for the chess-llm-core library.

    Controls default behaviours that apply across all modules.  Individual
    method calls can override these on a per-request basis.

    Attributes:
        default_provider:           Provider used when none is specified.
        default_tier:               Model tier used when none is specified.
        anthropic / openai / local: Per-provider connection settings.
        track_costs:                Whether to record cost data per request.
        cost_warning_threshold_usd: Log a warning when a single request
                                    exceeds this USD amount.
        log_requests:               Log outgoing prompts (may contain user
                                    data -- enable only in development).
        log_responses:              Log LLM response content (may contain
                                    user data -- enable only in development).
        requests_per_minute:        Client-side rate-limit ceiling.
        tokens_per_minute:          Client-side token-rate ceiling.
    """

    # -- Defaults -----------------------------------------------------------
    default_provider: str = "anthropic"
    default_tier: ModelTier = ModelTier.STANDARD

    # -- Provider-specific settings -----------------------------------------
    anthropic: ProviderSettings = field(default_factory=ProviderSettings)
    openai: ProviderSettings = field(default_factory=ProviderSettings)
    local: ProviderSettings = field(default_factory=ProviderSettings)

    # -- Cost tracking ------------------------------------------------------
    track_costs: bool = True
    cost_warning_threshold_usd: float = 1.0

    # -- Logging (disabled by default to protect user data) -----------------
    log_requests: bool = False
    log_responses: bool = False

    # -- Rate limiting ------------------------------------------------------
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_environment(cls) -> "Settings":
        """
        Build a ``Settings`` instance from environment variables.

        Variables that are unset or empty are silently skipped, leaving the
        dataclass defaults in place.  This is the default initialisation
        path used by ``get_settings()``.

        Returns:
            A fully-populated ``Settings`` object.
        """
        settings = cls()

        # -- Global ---------------------------------------------------------
        if provider := os.getenv("CHESS_LLM_DEFAULT_PROVIDER"):
            settings.default_provider = provider.lower()

        if tier := os.getenv("CHESS_LLM_DEFAULT_TIER"):
            try:
                settings.default_tier = ModelTier(tier.lower())
            except ValueError:
                pass  # Keep default if the value is unrecognised

        if track := os.getenv("CHESS_LLM_TRACK_COSTS"):
            settings.track_costs = track.lower() in ("true", "1", "yes")

        if log_req := os.getenv("CHESS_LLM_LOG_REQUESTS"):
            settings.log_requests = log_req.lower() in ("true", "1", "yes")

        if log_resp := os.getenv("CHESS_LLM_LOG_RESPONSES"):
            settings.log_responses = log_resp.lower() in ("true", "1", "yes")

        # -- Anthropic ------------------------------------------------------
        settings.anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_base := os.getenv("ANTHROPIC_API_BASE"):
            settings.anthropic.api_base_url = anthropic_base
        if anthropic_model := os.getenv("CHESS_LLM_ANTHROPIC_DEFAULT_MODEL"):
            settings.anthropic.default_model = anthropic_model

        # -- OpenAI ---------------------------------------------------------
        settings.openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai_base := os.getenv("OPENAI_API_BASE"):
            settings.openai.api_base_url = openai_base
        if openai_model := os.getenv("CHESS_LLM_OPENAI_DEFAULT_MODEL"):
            settings.openai.default_model = openai_model

        # -- Local (Ollama) -------------------------------------------------
        if local_base := os.getenv("OLLAMA_HOST"):
            settings.local.api_base_url = local_base
        if local_model := os.getenv("CHESS_LLM_LOCAL_DEFAULT_MODEL"):
            settings.local.default_model = local_model

        return settings

    # -----------------------------------------------------------------------
    # Provider look-up
    # -----------------------------------------------------------------------

    def get_provider_settings(self, provider: str) -> ProviderSettings:
        """
        Return the ``ProviderSettings`` for the given provider name.

        Args:
            provider: One of ``"anthropic"``, ``"openai"``, or ``"local"``.

        Raises:
            ValueError: If the provider name is not recognised.
        """
        provider = provider.lower()
        if provider not in _PROVIDER_MAP:
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Available: {sorted(_PROVIDER_MAP)}"
            )
        return getattr(self, provider)

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Return the API key for *provider*, or ``None`` if unset.

        Args:
            provider: Provider name (e.g. ``"anthropic"``).
        """
        return self.get_provider_settings(provider).api_key


# ---------------------------------------------------------------------------
# Global singleton management
# ---------------------------------------------------------------------------

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Return the global ``Settings`` singleton.

    On first call the settings are loaded from environment variables via
    ``Settings.from_environment()``.  Subsequent calls return the cached
    instance.  Use ``configure_settings()`` or ``reset_settings()`` to
    change the active configuration.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_environment()
    return _settings


def configure_settings(settings: Settings) -> None:
    """
    Replace the global ``Settings`` singleton.

    Args:
        settings: A pre-configured ``Settings`` instance.
    """
    global _settings
    _settings = settings


def reset_settings() -> None:
    """
    Discard the cached settings so the next ``get_settings()`` call
    re-reads from the environment.
    """
    global _settings
    _settings = None
