"""
Mentor coaching insights prompt template.

Generates personalized coaching insights based on comprehensive player analysis.
This is the main coaching prompt used in report generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chess_llm.config.models import ModelTier
from chess_llm.prompts.base import PromptTemplate, PromptVersion, OutputFormat


@dataclass
class MentorInsightsPrompt(PromptTemplate):
    """
    Generate mentor-style coaching insights.

    This prompt analyzes comprehensive player data and produces personalized
    coaching advice including:
    - Playing style analysis
    - Key improvement opportunities
    - Specific warnings about bad habits
    - Actionable next steps

    Recommended tier: STANDARD (requires good reasoning for personalized advice)
    """

    prompt_id: str = "mentor_insights"
    version: PromptVersion = field(default_factory=lambda: PromptVersion(1, 0, 0, "Initial version"))
    recommended_tier: ModelTier = ModelTier.STANDARD
    output_format: OutputFormat = OutputFormat.MARKDOWN
    estimated_input_tokens: int = 1500
    estimated_output_tokens: int = 1500

    # Required parameters
    username: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    # Optional enhanced data
    progression: Optional[Dict[str, Any]] = None
    behavioral_patterns: Optional[Dict[str, Any]] = None
    trouble_spots: Optional[List[Dict[str, Any]]] = None
    deep_chess960: Optional[Dict[str, Any]] = None
    recent_month_analysis: Optional[Dict[str, Any]] = None
    time_control_errors: Optional[Dict[str, Any]] = None
    opponent_style: Optional[Dict[str, Any]] = None
    platform: str = "chess.com"

    def render(self) -> str:
        """Render the complete mentor insights prompt."""
        # Calculate derived stats
        total_games = self.stats.get("total_games", 0)
        white_games = self.stats.get("white_games", 1)
        black_games = self.stats.get("black_games", 1)
        white_wins = self.stats.get("white_wins", 0)
        black_wins = self.stats.get("black_wins", 0)

        white_win_rate = round(white_wins / max(white_games, 1) * 100, 1)
        black_win_rate = round(black_wins / max(black_games, 1) * 100, 1)

        # Build context sections
        deep_960_text = self._format_chess960_insights()
        recent_changes_text = self._format_recent_changes()
        tc_errors_text = self._format_time_control_errors()
        opp_style_text = self._format_opponent_style()

        # Format progression data
        progression = self.progression or {}
        trouble_spots = self.trouble_spots or []

        prompt = f"""You are an experienced chess mentor having a one-on-one coaching session with {self.username}.
You've performed a DEEP analysis of their last 6 months of games. Share your most important observations.

IMPORTANT GUIDELINES:
- Write as a mentor speaking directly to the player (use "you")
- Focus on INSIGHTS they wouldn't see themselves, not just data recitation
- When discussing Chess960, focus on what it reveals about their fundamental chess understanding vs opening memorization
- Be specific about what has changed recently and whether they're experimenting with new things
- Call out specific weaknesses in certain time controls if the data shows them
- Discuss how they perform against different opponent styles
- Be encouraging but honest - point out real issues
- This is about understanding their PLAYING STYLE, not just statistics

PLAYER DATA:

**Recent Performance:**
- Total games: {total_games} in last 6 months
- Overall: {self.stats.get('wins', 0)}W / {self.stats.get('losses', 0)}L / {self.stats.get('draws', 0)}D
- As White: {white_win_rate}% win rate
- As Black: {black_win_rate}% win rate

**Progression (Recent Month vs Previous 5):**
- Recent month win rate: {progression.get('recent_win_rate', 'N/A')}%
- Earlier months win rate: {progression.get('earlier_win_rate', 'N/A')}%
- Change: {progression.get('win_rate_change', 0):+.1f}%
- Trend: {progression.get('overall_trend', 'stable')}
{recent_changes_text}

**Playing Style Indicators (from how games end):**
- Time pressure issues: {self._get_behavioral('time_pressure', 'None detected')}
- Tactical awareness: {self._get_behavioral('tactical_blindness', 'Good')}
- Fighting spirit: {self._get_behavioral('fighting_spirit', 'Average')}
- Finishing ability: {self._get_behavioral('finishing_ability', 'Average')}

**Openings Giving Trouble (< 40% win rate, 5+ games):**
{self._format_trouble_spots(trouble_spots)}

**Chess960 vs Standard Chess Analysis:**
{deep_960_text}
{tc_errors_text}
{opp_style_text}

**Win/Loss Methods Summary:**
- Wins by checkmate: {self.stats.get('win_methods', {}).get('opponent_checkmated', 0)}
- Wins by resignation: {self.stats.get('win_methods', {}).get('opponent_resigned', 0)}
- Wins on time: {self.stats.get('win_methods', {}).get('opponent_timeout', 0)}
- Losses by checkmate: {self.stats.get('loss_methods', {}).get('checkmated', 0)}
- Losses by resignation: {self.stats.get('loss_methods', {}).get('resigned', 0)}
- Losses on time: {self.stats.get('loss_methods', {}).get('timeout', 0)}

Write a mentor coaching message with these sections (use these exact headers):

## What I See in Your Games

(2-3 paragraphs analyzing their playing STYLE. Include:
- Are they opening-dependent or do they have solid fundamentals? (Use Chess960 data!)
- What has changed in the recent month? Are they experimenting?
- How do they handle different game speeds?
- Do they struggle against certain opponent types?)

## Your Biggest Opportunity

(1-2 paragraphs on THE single most impactful thing they should work on. Be specific - not "practice tactics" but WHY and HOW based on their actual patterns. Reference specific data.)

## Watch Out For

(1 paragraph on a potential bad habit or blind spot. Reference specific patterns from the data - time control issues, opponent style weaknesses, etc.)

## Your Path Forward

(3-4 bullet points of specific, actionable next steps for the coming month. Include at least one related to their time control weaknesses and one related to their Chess960/fundamental skills gap if applicable.)

Keep the entire response under 600 words. Be a mentor, not a statistician."""

        return prompt

    def render_system_prompt(self) -> Optional[str]:
        """Provide system context for the mentor role."""
        return (
            "You are an experienced chess coach providing personalized mentoring. "
            "You have deep expertise in chess strategy, tactics, and player development. "
            "Your coaching style is encouraging but honest, focusing on actionable insights."
        )

    def _get_behavioral(self, key: str, default: str) -> str:
        """Get a behavioral pattern value."""
        if self.behavioral_patterns:
            return self.behavioral_patterns.get(key, default)
        return default

    def _format_trouble_spots(self, spots: List[Dict[str, Any]]) -> str:
        """Format trouble spots as bullet list."""
        if not spots:
            return "- None detected"
        lines = []
        for spot in spots[:5]:
            lines.append(
                f"- {spot.get('name', 'Unknown')}: {spot.get('win_rate', 0)}% "
                f"({spot.get('games', 0)} games) - {spot.get('severity', 'unknown')}"
            )
        return "\n".join(lines)

    def _format_chess960_insights(self) -> str:
        """Format Chess960 analysis section."""
        if not self.deep_chess960 or self.deep_chess960.get("insufficient_data"):
            return "- No Chess960 data available"

        c960 = self.deep_chess960.get("chess960", {})
        std = self.deep_chess960.get("standard", {})
        comps = self.deep_chess960.get("comparisons", {})
        deep_insights = self.deep_chess960.get("deep_insights", [])

        text = f"""- Chess960: {c960.get('total', 0)} games, {c960.get('win_rate', 0)}% win rate
- Standard: {std.get('win_rate', 0)}% win rate
- Win rate difference: {comps.get('win_rate_diff', 0):+.1f}% (positive = better at 960)"""

        if deep_insights:
            text += "\n**CRITICAL Chess960 vs Standard Analysis:**"
            for insight in deep_insights[:3]:
                text += f"\n- {insight.get('finding', '')}"
                text += f"\n  → Implication: {insight.get('implication', '')}"
                text += f"\n  → Recommendation: {insight.get('recommendation', '')}"

        return text

    def _format_recent_changes(self) -> str:
        """Format recent month changes section."""
        if not self.recent_month_analysis or self.recent_month_analysis.get("insufficient_data"):
            return ""

        changes = self.recent_month_analysis.get("changes", [])
        if not changes:
            return ""

        text = f"\n**What Changed This Month ({self.recent_month_analysis.get('recent_month', 'recent')}):**"
        for change in changes[:4]:
            text += f"\n- [{change.get('type', '')}] {change.get('detail', '')}"
        return text

    def _format_time_control_errors(self) -> str:
        """Format time control error patterns."""
        if not self.time_control_errors:
            return ""

        by_speed = self.time_control_errors.get("by_speed", {})
        comparisons = self.time_control_errors.get("comparisons", [])

        if not comparisons and not by_speed:
            return ""

        text = "\n**Time Control Analysis:**"
        for comp in comparisons[:2]:
            text += f"\n- {comp.get('insight', '')}"

        for speed, data in by_speed.items():
            patterns = data.get("patterns", [])
            for p in patterns[:1]:
                text += f"\n- In {speed}: {p.get('description', '')}"

        return text

    def _format_opponent_style(self) -> str:
        """Format opponent style insights."""
        if not self.opponent_style or self.opponent_style.get("insufficient_data"):
            return ""

        patterns = self.opponent_style.get("patterns", [])
        if not patterns:
            return ""

        text = "\n**How You Perform Against Different Opponent Types:**"
        for p in patterns[:3]:
            text += f"\n- {p.get('detail', '')}"
        return text

    def validate(self) -> List[str]:
        """Validate prompt parameters."""
        errors = []
        if not self.username:
            errors.append("username is required")
        if not self.stats:
            errors.append("stats dictionary is required")
        if not self.stats.get("total_games"):
            errors.append("stats must contain total_games")
        return errors
