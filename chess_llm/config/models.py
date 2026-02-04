"""
Model configurations and tier definitions.

This module defines:
- Model tiers (CHEAP, STANDARD, PREMIUM) for cost optimization
- Per-model configurations including pricing, capabilities, and limits
- Helper functions to select appropriate models

Model Tiers:
- CHEAP: Fast, inexpensive models for simple tasks (extraction, classification)
- STANDARD: Balanced models for most coaching tasks
- PREMIUM: Best quality for complex analysis and key reports

Usage:
    from chess_llm.config.models import ModelTier, get_model_config

    # Get the default model for a tier
    config = get_model_config(ModelTier.STANDARD)
    print(f"Using {config.display_name} at ${config.input_cost_per_million}/M tokens")

    # Get a specific model
    config = get_model_config(ModelTier.CHEAP, provider="anthropic")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class ModelTier(Enum):
    """
    Model tiers for cost/quality optimization.

    Choose the appropriate tier based on task requirements:
    - CHEAP: High speed, low cost. Use for extraction, classification, simple summaries.
    - STANDARD: Good balance. Use for most coaching insights, opening analysis.
    - PREMIUM: Best quality. Use for comprehensive reports, complex strategic analysis.
    """

    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    # Identifiers
    model_id: str
    provider: str
    display_name: str
    tier: ModelTier

    # Capabilities
    max_context_tokens: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_function_calling: bool = False

    # Pricing (per million tokens, in USD)
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    # Performance characteristics
    avg_latency_ms: float = 1000.0  # Approximate average latency
    quality_score: float = 0.8  # Relative quality (0-1)

    # Recommended use cases
    recommended_for: List[str] = None

    def __post_init__(self):
        if self.recommended_for is None:
            self.recommended_for = []

    @property
    def cost_per_1k_tokens(self) -> Tuple[float, float]:
        """Return cost per 1K tokens (input, output)."""
        return (
            self.input_cost_per_million / 1000,
            self.output_cost_per_million / 1000,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost


# =============================================================================
# Anthropic Models
# =============================================================================

ANTHROPIC_MODELS: Dict[str, ModelConfig] = {
    # Claude 3.5 Haiku - Fast and cheap
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
    # Claude 3.5 Sonnet - Balanced performance
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
    # Claude 3 Opus - Highest quality (deprecated but included for reference)
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

# =============================================================================
# OpenAI Models (stubs for future implementation)
# =============================================================================

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

# =============================================================================
# Local Models (stubs for future implementation)
# =============================================================================

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
        input_cost_per_million=0.0,  # Free (local)
        output_cost_per_million=0.0,
        avg_latency_ms=800,
        quality_score=0.65,
        recommended_for=[
            "simple_extraction",
            "offline_analysis",
        ],
    ),
}

# =============================================================================
# All Models Registry
# =============================================================================

ALL_MODELS: Dict[str, ModelConfig] = {
    **ANTHROPIC_MODELS,
    **OPENAI_MODELS,
    **LOCAL_MODELS,
}

# Default models per tier per provider
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


def get_model_config(
    tier_or_model: Union[ModelTier, str],
    provider: str = "anthropic",
) -> Optional[ModelConfig]:
    """
    Get model configuration by tier or model ID.

    Args:
        tier_or_model: Either a ModelTier enum or a specific model ID string
        provider: Provider to use when looking up by tier (default: anthropic)

    Returns:
        ModelConfig for the requested model, or None if not found
    """
    # If it's a model ID string, look it up directly
    if isinstance(tier_or_model, str):
        return ALL_MODELS.get(tier_or_model)

    # If it's a tier, get the default model for that tier and provider
    if isinstance(tier_or_model, ModelTier):
        if provider not in DEFAULT_MODELS:
            return None
        if tier_or_model not in DEFAULT_MODELS[provider]:
            return None

        model_id = DEFAULT_MODELS[provider][tier_or_model]
        return ALL_MODELS.get(model_id)

    return None


def get_default_model(provider: str, tier: ModelTier) -> Optional[str]:
    """
    Get the default model ID for a provider and tier.

    Args:
        provider: Provider name (e.g., 'anthropic')
        tier: Model tier

    Returns:
        Model ID string, or None if not found
    """
    if provider not in DEFAULT_MODELS:
        return None
    return DEFAULT_MODELS[provider].get(tier)


def get_models_by_tier(tier: ModelTier) -> List[Tuple[str, ModelConfig]]:
    """
    Get all models for a specific tier.

    Args:
        tier: The model tier to filter by

    Returns:
        List of (model_id, ModelConfig) tuples matching the tier
    """
    return [(model_id, config) for model_id, config in ALL_MODELS.items() if config.tier == tier]


def get_models_for_tier(tier: ModelTier, provider: Optional[str] = None) -> List[ModelConfig]:
    """
    Get all models for a specific tier.

    Args:
        tier: The model tier to filter by
        provider: Optional provider to filter by

    Returns:
        List of ModelConfig objects matching the criteria
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
    Get the cheapest model meeting the specified requirements.

    Args:
        min_quality: Minimum quality score (0-1)
        requires_vision: Whether vision support is required
        provider: Optional provider filter

    Returns:
        The cheapest ModelConfig meeting requirements

    Raises:
        ValueError: If no model meets requirements
    """
    candidates = list(ALL_MODELS.values())

    if provider:
        candidates = [m for m in candidates if m.provider == provider]
    if requires_vision:
        candidates = [m for m in candidates if m.supports_vision]
    candidates = [m for m in candidates if m.quality_score >= min_quality]

    if not candidates:
        raise ValueError("No models meet the specified requirements")

    # Sort by total cost (assuming 1:1 input/output ratio for simplicity)
    candidates.sort(
        key=lambda m: m.input_cost_per_million + m.output_cost_per_million
    )

    return candidates[0]
