"""
Model configurations and tier definitions for chess-llm-core.

This module is the single source of truth for every LLM model the library
can use.  It defines:

    - **Model tiers** (CHEAP / STANDARD / PREMIUM) for cost-quality routing
    - **Per-model configurations** including pricing, capabilities, and limits
    - **Lookup helpers** to select the right model for a given task

Model Tiers:
    CHEAP:    Fast, inexpensive models for simple tasks (extraction,
              classification, formatting).
    STANDARD: Balanced quality/cost for most coaching tasks (insights,
              opening analysis, scouting reports).
    PREMIUM:  Highest quality for complex analysis and key reports.

Usage::

    from chess_llm.config.models import ModelTier, get_model_config

    config = get_model_config(ModelTier.STANDARD)
    print(f"Using {config.display_name} at ${config.input_cost_per_million}/M tokens")

    config = get_model_config("claude-3-5-haiku-20241022")

Maintenance:
    To add a new model, create a ``ModelConfig`` entry in the appropriate
    ``*_MODELS`` dict and add it to ``DEFAULT_MODELS`` if it should be the
    default for a tier/provider combination.  The ``ALL_MODELS`` registry
    is built automatically from the provider dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class ModelTier(Enum):
    """
    Model tiers for cost/quality optimisation.

    Choose the appropriate tier based on task requirements:

    - ``CHEAP``:    High speed, low cost.  Use for extraction,
                    classification, and simple summaries.
    - ``STANDARD``: Good balance.  Use for coaching insights, opening
                    analysis, and scouting reports.
    - ``PREMIUM``:  Best quality.  Use for comprehensive reports and
                    complex strategic analysis.
    """

    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """
    Immutable configuration for a specific LLM model.

    Each model in the registry is represented by one ``ModelConfig`` that
    captures its identifier, capabilities, pricing, and recommended use
    cases.

    Attributes:
        model_id:                API-level model identifier string.
        provider:                Provider that serves this model.
        display_name:            Human-readable label for UI/logs.
        tier:                    Cost/quality tier.
        max_context_tokens:      Maximum input context window (tokens).
        max_output_tokens:       Maximum tokens the model can generate.
        supports_vision:         Whether image inputs are accepted.
        supports_function_calling: Whether tool/function calling is supported.
        input_cost_per_million:  USD cost per 1 M input tokens.
        output_cost_per_million: USD cost per 1 M output tokens.
        avg_latency_ms:          Approximate average response latency.
        quality_score:           Relative quality rating (0.0 -- 1.0).
        recommended_for:         List of task labels this model excels at.
    """

    # -- Identifiers --------------------------------------------------------
    model_id: str
    provider: str
    display_name: str
    tier: ModelTier

    # -- Capabilities -------------------------------------------------------
    max_context_tokens: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_function_calling: bool = False

    # -- Pricing (per million tokens, USD) ----------------------------------
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    # -- Performance --------------------------------------------------------
    avg_latency_ms: float = 1000.0
    quality_score: float = 0.8  # 0.0 = worst, 1.0 = best

    # -- Recommended use cases ----------------------------------------------
    recommended_for: List[str] = None

    def __post_init__(self) -> None:
        if self.recommended_for is None:
            self.recommended_for = []

    @property
    def cost_per_1k_tokens(self) -> Tuple[float, float]:
        """Return ``(input, output)`` cost per 1 000 tokens."""
        return (
            self.input_cost_per_million / 1000,
            self.output_cost_per_million / 1000,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the USD cost for a request of the given size.

        Args:
            input_tokens:  Number of prompt / context tokens.
            output_tokens: Number of generated tokens.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost


# ==========================================================================
# Model registries -- one dict per provider
# ==========================================================================

ANTHROPIC_MODELS: Dict[str, ModelConfig] = {
    "claude-3-5-haiku-20241022": ModelConfig(
        model_id="claude-3-5-haiku-20241022",
        provider="anthropic",
        display_name="Claude 3.5 Haiku",
        tier=ModelTier.CHEAP,
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_million=1.00,
        output_cost_per_million=5.00,
        avg_latency_ms=500,
        quality_score=0.75,
        recommended_for=[
            "key_area_extraction",
            "simple_classification",
            "quick_summaries",
            "data_formatting",
        ],
    ),
    "claude-sonnet-4-20250514": ModelConfig(
        model_id="claude-sonnet-4-20250514",
        provider="anthropic",
        display_name="Claude Sonnet 4",
        tier=ModelTier.STANDARD,
        max_context_tokens=200_000,
        max_output_tokens=16_000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        avg_latency_ms=1500,
        quality_score=0.90,
        recommended_for=[
            "coaching_insights",
            "opening_analysis",
            "tactical_explanations",
            "improvement_plans",
            "scouting_reports",
        ],
    ),
    # Claude 3 Opus -- highest quality (legacy, kept for reference/fallback)
    "claude-3-opus-20240229": ModelConfig(
        model_id="claude-3-opus-20240229",
        provider="anthropic",
        display_name="Claude 3 Opus",
        tier=ModelTier.PREMIUM,
        max_context_tokens=200_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
        avg_latency_ms=3000,
        quality_score=0.95,
        recommended_for=[
            "comprehensive_reports",
            "complex_strategic_analysis",
            "personalized_improvement_plans",
        ],
    ),
}

OPENAI_MODELS: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        model_id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        tier=ModelTier.STANDARD,
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
        avg_latency_ms=1200,
        quality_score=0.88,
        recommended_for=[
            "coaching_insights",
            "opening_analysis",
        ],
    ),
    "gpt-4o-mini": ModelConfig(
        model_id="gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o Mini",
        tier=ModelTier.CHEAP,
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        avg_latency_ms=600,
        quality_score=0.72,
        recommended_for=[
            "simple_extraction",
            "data_formatting",
        ],
    ),
}

LOCAL_MODELS: Dict[str, ModelConfig] = {
    "qwen2.5-7b": ModelConfig(
        model_id="qwen2.5-7b",
        provider="local",
        display_name="Qwen 2.5 7B",
        tier=ModelTier.CHEAP,
        max_context_tokens=32_000,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_function_calling=True,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        avg_latency_ms=800,
        quality_score=0.65,
        recommended_for=[
            "simple_extraction",
            "offline_analysis",
        ],
    ),
}

# ==========================================================================
# Unified registry (built automatically from provider dicts)
# ==========================================================================

ALL_MODELS: Dict[str, ModelConfig] = {
    **ANTHROPIC_MODELS,
    **OPENAI_MODELS,
    **LOCAL_MODELS,
}

# Default model ID for each (provider, tier) combination.
DEFAULT_MODELS: Dict[str, Dict[ModelTier, str]] = {
    "anthropic": {
        ModelTier.CHEAP: "claude-3-5-haiku-20241022",
        ModelTier.STANDARD: "claude-sonnet-4-20250514",
        ModelTier.PREMIUM: "claude-3-opus-20240229",
    },
    "openai": {
        ModelTier.CHEAP: "gpt-4o-mini",
        ModelTier.STANDARD: "gpt-4o",
        ModelTier.PREMIUM: "gpt-4o",
    },
    "local": {
        ModelTier.CHEAP: "qwen2.5-7b",
        ModelTier.STANDARD: "qwen2.5-7b",
        ModelTier.PREMIUM: "qwen2.5-7b",
    },
}


# ==========================================================================
# Public look-up helpers
# ==========================================================================


def get_model_config(
    tier_or_model: Union[ModelTier, str],
    provider: str = "anthropic",
) -> Optional[ModelConfig]:
    """
    Get model configuration by tier or explicit model ID.

    Args:
        tier_or_model: A ``ModelTier`` enum (resolves via ``DEFAULT_MODELS``)
                       or a model ID string (looked up directly).
        provider:      Provider to use when resolving by tier.

    Returns:
        The matching ``ModelConfig``, or ``None`` if not found.
    """
    if isinstance(tier_or_model, str):
        return ALL_MODELS.get(tier_or_model)

    if isinstance(tier_or_model, ModelTier):
        provider_defaults = DEFAULT_MODELS.get(provider)
        if provider_defaults is None:
            return None
        model_id = provider_defaults.get(tier_or_model)
        if model_id is None:
            return None
        return ALL_MODELS.get(model_id)

    return None


def get_default_model(provider: str, tier: ModelTier) -> Optional[str]:
    """
    Return the default model ID for a provider + tier combination.

    Args:
        provider: Provider name (e.g. ``"anthropic"``).
        tier:     Desired model tier.

    Returns:
        Model ID string, or ``None`` if the combination is unknown.
    """
    provider_defaults = DEFAULT_MODELS.get(provider)
    if provider_defaults is None:
        return None
    return provider_defaults.get(tier)


def get_models_by_tier(tier: ModelTier) -> List[Tuple[str, ModelConfig]]:
    """
    Return all ``(model_id, ModelConfig)`` pairs that belong to *tier*.

    Args:
        tier: The tier to filter by.
    """
    return [
        (model_id, config)
        for model_id, config in ALL_MODELS.items()
        if config.tier == tier
    ]


def get_models_for_tier(
    tier: ModelTier,
    provider: Optional[str] = None,
) -> List[ModelConfig]:
    """
    Return all ``ModelConfig`` objects for *tier*, optionally filtered by
    *provider*.

    Args:
        tier:     The tier to filter by.
        provider: If given, only return models from this provider.
    """
    models = [m for m in ALL_MODELS.values() if m.tier == tier]
    if provider:
        models = [m for m in models if m.provider == provider]
    return models


def get_cheapest_model(
    min_quality: float = 0.6,
    requires_vision: bool = False,
    provider: Optional[str] = None,
) -> ModelConfig:
    """
    Find the cheapest model that meets the given requirements.

    Models are ranked by total cost (assuming a 1:1 input/output ratio).

    Args:
        min_quality:    Minimum acceptable ``quality_score`` (0.0 -- 1.0).
        requires_vision: If ``True``, exclude models without vision support.
        provider:        Optional provider filter.

    Returns:
        The cheapest qualifying ``ModelConfig``.

    Raises:
        ValueError: If no model satisfies all constraints.
    """
    candidates = list(ALL_MODELS.values())

    if provider:
        candidates = [m for m in candidates if m.provider == provider]
    if requires_vision:
        candidates = [m for m in candidates if m.supports_vision]
    candidates = [m for m in candidates if m.quality_score >= min_quality]

    if not candidates:
        raise ValueError("No models meet the specified requirements")

    candidates.sort(
        key=lambda m: m.input_cost_per_million + m.output_cost_per_million
    )
    return candidates[0]
