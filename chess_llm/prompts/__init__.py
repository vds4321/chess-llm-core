"""
Versioned prompt templates for chess coaching.

Prompts are organized by category:
- coaching: Mentor insights, improvement plans, tactical advice
- scouting: Battle plans, opponent analysis
- extraction: Key area extraction, data parsing

Each prompt template includes:
- Version tracking for A/B testing
- Recommended model tier
- Token budget estimates
- Structured output formats

Usage:
    from chess_llm.prompts import MentorInsightsPrompt, ModelTier

    # Create a prompt
    prompt = MentorInsightsPrompt(
        username="player123",
        stats=player_stats,
        progression=progression_data,
    )

    # Get the rendered prompt
    text = prompt.render()

    # Get recommended configuration
    tier = prompt.recommended_tier  # ModelTier.STANDARD
    max_tokens = prompt.estimated_output_tokens  # ~1500
"""

from chess_llm.prompts.base import PromptTemplate, PromptVersion
from chess_llm.prompts.coaching.mentor_insights import MentorInsightsPrompt
from chess_llm.prompts.coaching.opening_analysis import OpeningAnalysisPrompt
from chess_llm.prompts.extraction.key_areas import KeyAreasExtractionPrompt
from chess_llm.prompts.scouting.battle_plan import BattlePlanPrompt

__all__ = [
    # Base
    "PromptTemplate",
    "PromptVersion",
    # Coaching
    "MentorInsightsPrompt",
    "OpeningAnalysisPrompt",
    # Extraction
    "KeyAreasExtractionPrompt",
    # Scouting
    "BattlePlanPrompt",
]
