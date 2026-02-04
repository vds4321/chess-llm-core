"""
Provider registry for managing LLM providers.

The registry allows:
- Getting providers by name or tier
- Registering custom providers
- Listing available providers

Usage:
    from chess_llm.providers import get_provider
    from chess_llm.config.models import ModelTier

    # Get provider by name (uses default model)
    provider = get_provider("anthropic")

    # Get provider for a specific tier
    provider = get_provider("anthropic", tier=ModelTier.CHEAP)

    # Get provider with specific model
    provider = get_provider("anthropic", model_id="claude-3-5-haiku-20241022")
"""

from typing import Any, Callable, Dict, List, Optional, Type

from chess_llm.providers.base import LLMProvider, BaseLLMProvider
from chess_llm.config.models import ModelTier
from chess_llm.config.settings import get_settings


# Type for provider factory functions
ProviderFactory = Callable[..., LLMProvider]

# Registry of provider factories
_providers: Dict[str, ProviderFactory] = {}


def register_provider(
    provider_id: str,
    factory: ProviderFactory,
    override: bool = False,
) -> None:
    """
    Register a provider factory.

    Args:
        provider_id: Unique identifier for the provider (e.g., 'anthropic')
        factory: Factory function or class that creates provider instances
        override: If True, allows overriding existing registrations

    Raises:
        ValueError: If provider already registered and override=False
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
    Get a provider instance.

    Args:
        provider_id: Provider identifier (e.g., 'anthropic'). Uses settings default if not provided.
        tier: Model tier to use. Ignored if model_id is specified.
        model_id: Specific model to use. Overrides tier.
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider not found or not available
    """
    # Use settings default if no provider specified
    if provider_id is None:
        settings = get_settings()
        provider_id = settings.default_provider

    # Ensure provider is registered
    _ensure_default_providers_registered()

    if provider_id not in _providers:
        available = list(_providers.keys())
        raise ValueError(
            f"Unknown provider: '{provider_id}'. "
            f"Available providers: {available}"
        )

    factory = _providers[provider_id]

    # Build kwargs for the factory
    factory_kwargs = dict(kwargs)
    if tier is not None:
        factory_kwargs["tier"] = tier
    if model_id is not None:
        factory_kwargs["model_id"] = model_id

    return factory(**factory_kwargs)


def get_provider_for_tier(
    tier: ModelTier,
    provider_id: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Get a provider configured for a specific tier.

    Convenience function for getting tier-based providers.

    Args:
        tier: Model tier (CHEAP, STANDARD, PREMIUM)
        provider_id: Provider to use. Uses settings default if not provided.
        **kwargs: Additional provider arguments.

    Returns:
        LLMProvider instance configured for the tier
    """
    return get_provider(provider_id=provider_id, tier=tier, **kwargs)


def list_providers() -> List[str]:
    """
    List all registered provider IDs.

    Returns:
        List of provider identifiers
    """
    _ensure_default_providers_registered()
    return list(_providers.keys())


def is_provider_available(provider_id: str) -> bool:
    """
    Check if a provider is available (registered and dependencies installed).

    Args:
        provider_id: Provider identifier to check

    Returns:
        True if provider is available
    """
    _ensure_default_providers_registered()

    if provider_id not in _providers:
        return False

    # Try to instantiate to check if dependencies are available
    try:
        # For anthropic, check if the package is installed
        if provider_id == "anthropic":
            import anthropic
            return True
        elif provider_id == "openai":
            import openai
            return True
        elif provider_id == "local":
            # Local provider doesn't require external packages
            return True
        return True
    except ImportError:
        return False


def _ensure_default_providers_registered() -> None:
    """Register default providers if not already registered."""
    if "anthropic" not in _providers:
        _register_anthropic()
    if "openai" not in _providers:
        _register_openai()
    if "local" not in _providers:
        _register_local()


def _register_anthropic() -> None:
    """Register Anthropic provider."""
    def factory(**kwargs: Any) -> LLMProvider:
        from chess_llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)

    register_provider("anthropic", factory)


def _register_openai() -> None:
    """Register OpenAI provider (stub)."""
    def factory(**kwargs: Any) -> LLMProvider:
        raise NotImplementedError(
            "OpenAI provider not yet implemented. "
            "Coming soon! Use 'anthropic' provider for now."
        )

    register_provider("openai", factory)


def _register_local() -> None:
    """Register local provider (stub)."""
    def factory(**kwargs: Any) -> LLMProvider:
        raise NotImplementedError(
            "Local provider not yet implemented. "
            "Coming soon! Use 'anthropic' provider for now."
        )

    register_provider("local", factory)
