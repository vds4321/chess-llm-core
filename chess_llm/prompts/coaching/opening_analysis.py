"""
Opening analysis prompt template.

Generates insights about specific opening performance and recommendations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chess_llm.config.models import ModelTier
from chess_llm.prompts.base import PromptTemplate, PromptVersion, OutputFormat


@dataclass
class OpeningAnalysisPrompt(PromptTemplate):
    """
    Generate opening-specific coaching insights.

    This prompt analyzes performance in a specific opening and provides
    targeted advice for improvement.

    Recommended tier: STANDARD (needs chess knowledge for good advice)
    """

    prompt_id: str = "opening_analysis"
    version: PromptVersion = field(default_factory=lambda: PromptVersion(1, 0, 0, "Initial version"))
    recommended_tier: ModelTier = ModelTier.STANDARD
    output_format: OutputFormat = OutputFormat.MARKDOWN
    estimated_input_tokens: int = 800
    estimated_output_tokens: int = 400

    # Required parameters
    opening_name: str = ""
    win_rate: float = 0.0
    games_count: int = 0
    player_color: str = "white"  # "white" or "black"

    # Optional context
    common_mistakes: Optional[List[str]] = None
    typical_positions: Optional[List[str]] = None  # FEN strings
    opponent_level: Optional[str] = None  # e.g., "similar", "higher", "lower"

    def render(self) -> str:
        """Render the opening analysis prompt."""
        color_desc = "as White" if self.player_color == "white" else "as Black"

        mistakes_text = ""
        if self.common_mistakes:
            mistakes_text = "\nCommon mistakes observed:\n" + "\n".join(
                f"- {m}" for m in self.common_mistakes[:5]
            )

        prompt = f"""Analyze performance in the {self.opening_name} {color_desc}.

Statistics:
- Games played: {self.games_count}
- Win rate: {self.win_rate}%
- Player color: {self.player_color.title()}
{f"- Typical opponent level: {self.opponent_level}" if self.opponent_level else ""}
{mistakes_text}

Provide a brief analysis (150-200 words) covering:
1. Is this win rate concerning, acceptable, or good for this opening?
2. One key idea or plan to focus on in this opening
3. A common pitfall to avoid
4. One specific suggestion for improvement

Be practical and actionable. Reference the opening's typical strategic themes."""

        return prompt

    def validate(self) -> List[str]:
        """Validate prompt parameters."""
        errors = []
        if not self.opening_name:
            errors.append("opening_name is required")
        if self.games_count < 1:
            errors.append("games_count must be at least 1")
        if self.player_color not in ("white", "black"):
            errors.append("player_color must be 'white' or 'black'")
        return errors
