"""
Provider registry for managing LLM provider instances.

The registry uses a **factory pattern** to decouple provider creation from
consumer code.  Providers are registered as lightweight factory callables
and instantiated on demand, which also means optional dependencies (like
the ``anthropic`` SDK) are only imported when actually needed.

Public API::

    from chess_llm.providers import get_provider
    from chess_llm.config.models import ModelTier

    # Default provider + default tier (from settings)
    provider = get_provider()

    # Explicit provider and tier
    provider = get_provider("anthropic", tier=ModelTier.CHEAP)

    # Explicit model ID
    provider = get_provider("anthropic", model_id="claude-haiku-4-5-20251001")

Extending with custom providers::

    from chess_llm.providers.registry import register_provider

    register_provider("my_provider", my_factory_function)

Maintenance:
    Built-in providers are lazily registered via
    ``_ensure_default_providers_registered()``.  To add a new built-in
    provider, write a ``_register_<name>()`` helper and call it from there.
"""

from typing import Any, Callable, Dict, List, Optional

from chess_llm.providers.base import LLMProvider, BaseLLMProvider
from chess_llm.config.models import ModelTier
from chess_llm.config.settings import get_settings

# Type alias for callables that create provider instances.
ProviderFactory = Callable[..., LLMProvider]

# Internal registry: provider_id -> factory callable
_providers: Dict[str, ProviderFactory] = {}


# ---------------------------------------------------------------------------
# Public registration / look-up functions
# ---------------------------------------------------------------------------


def register_provider(
    provider_id: str,
    factory: ProviderFactory,
    override: bool = False,
) -> None:
    """
    Register a provider factory under *provider_id*.

    Args:
        provider_id: Unique identifier (e.g. ``"anthropic"``).
        factory:     Callable that returns an ``LLMProvider`` instance.
        override:    Set ``True`` to replace an existing registration.

    Raises:
        ValueError: If *provider_id* is already registered and
                    *override* is ``False``.
    """
    if provider_id in _providers and not override:
        raise ValueError(
            f"Provider '{provider_id}' already registered. "
            "Use override=True to replace."
        )
    _providers[provider_id] = factory


def get_provider(
    provider_id: Optional[str] = None,
    tier: Optional[ModelTier] = None,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Create and return a provider instance.

    Resolution order for the provider name:
        1. Explicit *provider_id* argument.
        2. ``Settings.default_provider`` (from environment / config).

    Args:
        provider_id: Provider identifier.  Falls back to settings default.
        tier:        Model tier.  Ignored when *model_id* is set.
        model_id:    Explicit model.  Overrides *tier*.
        **kwargs:    Extra arguments forwarded to the provider constructor.

    Returns:
        A ready-to-use ``LLMProvider`` instance.

    Raises:
        ValueError: If the provider is unknown or unavailable.
    """
    if provider_id is None:
        settings = get_settings()
        provider_id = settings.default_provider

    _ensure_default_providers_registered()

    if provider_id not in _providers:
        available = list(_providers.keys())
        raise ValueError(
            f"Unknown provider: '{provider_id}'. "
            f"Available providers: {available}"
        )

    factory_kwargs: Dict[str, Any] = dict(kwargs)
    if tier is not None:
        factory_kwargs["tier"] = tier
    if model_id is not None:
        factory_kwargs["model_id"] = model_id

    return _providers[provider_id](**factory_kwargs)


def get_provider_for_tier(
    tier: ModelTier,
    provider_id: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Convenience wrapper: get a provider configured for a specific tier.

    Args:
        tier:        Desired model tier.
        provider_id: Provider to use (falls back to settings default).
        **kwargs:    Forwarded to ``get_provider()``.
    """
    return get_provider(provider_id=provider_id, tier=tier, **kwargs)


def list_providers() -> List[str]:
    """Return a list of all registered provider identifiers."""
    _ensure_default_providers_registered()
    return list(_providers.keys())


def is_provider_available(provider_id: str) -> bool:
    """
    Check whether *provider_id* is registered **and** its runtime
    dependencies are importable.

    Returns ``False`` for unknown providers or missing packages.
    """
    _ensure_default_providers_registered()

    if provider_id not in _providers:
        return False

    # Verify that the required SDK package is importable.
    _IMPORT_CHECK = {
        "anthropic": "anthropic",
        "openai": "openai",
    }
    package = _IMPORT_CHECK.get(provider_id)
    if package is None:
        return True  # No external dependency (e.g. "local")
    try:
        __import__(package)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Lazy registration of built-in providers
# ---------------------------------------------------------------------------


def _ensure_default_providers_registered() -> None:
    """Register the built-in providers on first access."""
    if "anthropic" not in _providers:
        _register_anthropic()
    if "openai" not in _providers:
        _register_openai()
    if "local" not in _providers:
        _register_local()


def _register_anthropic() -> None:
    """Register the Anthropic (Claude) provider factory."""

    def factory(**kwargs: Any) -> LLMProvider:
        from chess_llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)

    register_provider("anthropic", factory)


def _register_openai() -> None:
    """Register the OpenAI provider factory."""

    def factory(**kwargs: Any) -> LLMProvider:
        from chess_llm.providers.openai import OpenAIProvider

        return OpenAIProvider(**kwargs)

    register_provider("openai", factory)


def _register_local() -> None:
    """Register the local / Ollama provider factory (stub)."""

    def factory(**kwargs: Any) -> LLMProvider:
        raise NotImplementedError(
            "Local provider not yet implemented. "
            "Coming soon! Use 'anthropic' provider for now."
        )

    register_provider("local", factory)
