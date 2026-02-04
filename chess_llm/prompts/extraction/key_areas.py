"""
Key improvement areas extraction prompt.

Extracts structured data from coaching reports for progress tracking.
Uses CHEAP tier since this is primarily data extraction, not generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chess_llm.config.models import ModelTier
from chess_llm.prompts.base import PromptTemplate, PromptVersion, OutputFormat


@dataclass
class KeyAreasExtractionPrompt(PromptTemplate):
    """
    Extract key improvement areas from a coaching report.

    This prompt extracts structured data for future progress tracking,
    including main weaknesses, improvement areas, and behavioral patterns.

    Recommended tier: CHEAP (data extraction task, doesn't need complex reasoning)
    """

    prompt_id: str = "key_areas_extraction"
    version: PromptVersion = field(default_factory=lambda: PromptVersion(1, 0, 0, "Initial version"))
    recommended_tier: ModelTier = ModelTier.CHEAP
    output_format: OutputFormat = OutputFormat.JSON
    estimated_input_tokens: int = 2000
    estimated_output_tokens: int = 600

    # Required parameters
    report_content: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    # Maximum report length to include (truncate if longer)
    max_report_chars: int = 4000

    def render(self) -> str:
        """Render the key areas extraction prompt."""
        # Calculate stats for context
        total_games = self.stats.get("total_games", 0)
        wins = self.stats.get("wins", 0)
        white_wins = self.stats.get("white_wins", 0)
        white_games = self.stats.get("white_games", 1)
        black_wins = self.stats.get("black_wins", 0)
        black_games = self.stats.get("black_games", 1)

        win_rate = round(wins / max(total_games, 1) * 100, 1)
        white_win_rate = round(white_wins / max(white_games, 1) * 100, 1)
        black_win_rate = round(black_wins / max(black_games, 1) * 100, 1)

        # Truncate report if too long
        report_text = self.report_content[:self.max_report_chars]
        if len(self.report_content) > self.max_report_chars:
            report_text += "\n...[truncated]"

        primary_tc = self.stats.get("time_control_analysis", {}).get("primary_category", "Unknown")

        prompt = f"""Analyze this chess coaching report and extract the key improvement areas for future comparison.

REPORT:
{report_text}

PLAYER STATS AT TIME OF REPORT:
- Total games: {total_games}
- Win rate: {win_rate}%
- White win rate: {white_win_rate}%
- Black win rate: {black_win_rate}%
- Primary time control: {primary_tc}

Extract and return a JSON object with these fields:
{{
    "main_weakness": "The single biggest weakness identified (1 sentence)",
    "improvement_areas": ["area1", "area2", "area3"],
    "opening_issues": ["opening1", "opening2"],
    "behavioral_patterns": {{
        "time_pressure": "description of time pressure issues if any",
        "tactical_awareness": "assessment",
        "endgame_skill": "assessment if mentioned"
    }},
    "win_rate_at_report": {win_rate},
    "games_at_report": {total_games},
    "strengths": ["strength1", "strength2"],
    "recommended_focus": "The primary thing they should practice"
}}

Return ONLY valid JSON, no other text."""

        return prompt

    def validate(self) -> List[str]:
        """Validate prompt parameters."""
        errors = []
        if not self.report_content:
            errors.append("report_content is required")
        if not self.stats:
            errors.append("stats dictionary is required")
        return errors

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response with fallback handling."""
        import json

        text = response.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Remove "json" prefix if present
        if text.startswith("json"):
            text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Return minimal fallback data
            return {
                "win_rate_at_report": self.stats.get("wins", 0) / max(self.stats.get("total_games", 1), 1) * 100,
                "games_at_report": self.stats.get("total_games", 0),
                "extraction_failed": True,
            }
