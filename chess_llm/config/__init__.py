"""
Configuration module for chess-llm-core.

Provides:
- Model tier definitions (cheap, standard, premium)
- Model configurations with pricing
- Environment-based settings
"""

from chess_llm.config.models import (
    ModelTier,
    ModelConfig,
    get_model_config,
    get_default_model,
    get_models_by_tier,
    get_models_for_tier,
    ANTHROPIC_MODELS,
    OPENAI_MODELS,
)
from chess_llm.config.settings import Settings, get_settings

__all__ = [
    "ModelTier",
    "ModelConfig",
    "get_model_config",
    "get_default_model",
    "get_models_by_tier",
    "get_models_for_tier",
    "Settings",
    "get_settings",
    "ANTHROPIC_MODELS",
    "OPENAI_MODELS",
]
