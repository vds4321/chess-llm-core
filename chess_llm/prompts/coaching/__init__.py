"""
Coaching prompt templates -- personalised advice and analysis for players.

Templates:
    - ``MentorInsightsPrompt`` -- comprehensive coaching report.
    - ``OpeningAnalysisPrompt`` -- targeted advice for a single opening.
"""

from chess_llm.prompts.coaching.mentor_insights import MentorInsightsPrompt
from chess_llm.prompts.coaching.opening_analysis import OpeningAnalysisPrompt

__all__ = [
    "MentorInsightsPrompt",
    "OpeningAnalysisPrompt",
]
