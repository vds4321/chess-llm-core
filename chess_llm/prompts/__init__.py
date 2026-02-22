"""
Versioned prompt templates for chess coaching.

Prompts are organised by domain:

    - **coaching** -- Mentor insights, opening analysis, improvement plans.
    - **scouting** -- Battle plans and opponent analysis.
    - **extraction** -- Structured data extraction from free-text reports.

Each template includes:
    - Semantic version tracking (for A/B testing via ``kaspar_eval``).
    - Recommended model tier and token budget.
    - Output format declaration.
    - Optional response-parsing logic.

Usage::

    from chess_llm.prompts import MentorInsightsPrompt

    prompt = MentorInsightsPrompt(
        username="player123",
        stats=player_stats,
        progression=progression_data,
    )

    text = prompt.render()
    tier = prompt.recommended_tier
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
