"""Battle Plan prompt for scouting/opponent analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chess_llm.config.models import ModelTier
from chess_llm.prompts.base import PromptTemplate


@dataclass
class BattlePlanPrompt(PromptTemplate):
    """
    Prompt for generating a battle plan against a specific opponent.

    This analyzes the opponent's playing patterns and creates
    actionable strategies for how to beat them.

    Attributes:
        opponent_username: The opponent to scout
        player_username: The player receiving the battle plan
        opponent_stats: Dict with opponent's game statistics
        player_stats: Dict with player's own statistics
    """

    prompt_id: str = field(default="battle_plan", init=False)
    version: str = field(default="1.0.0", init=False)
    recommended_tier: ModelTier = field(default=ModelTier.STANDARD, init=False)

    opponent_username: str = ""
    player_username: str = ""
    opponent_stats: Dict[str, Any] = field(default_factory=dict)
    player_stats: Dict[str, Any] = field(default_factory=dict)

    def _calc_color_win_rate(self, stats: Dict[str, Any], color: str) -> float:
        """Calculate win rate for a specific color."""
        color_wins = stats.get(f"{color}_wins", 0)
        color_games = stats.get(f"{color}_games", 1)
        return round(color_wins / color_games * 100, 1) if color_games > 0 else 0.0

    def _format_openings_for_scouting(self, opening_stats: Dict[str, Any]) -> str:
        """Format opening statistics for the prompt."""
        if not opening_stats:
            return "- No opening data available"

        sorted_openings = sorted(
            opening_stats.items(),
            key=lambda x: (-x[1].get("total_games", 0), -x[1].get("win_rate", 0))
        )[:5]

        lines = []
        for name, data in sorted_openings:
            games = data.get("total_games", 0)
            win_rate = data.get("win_rate", 0)
            lines.append(f"- {name}: {games} games, {win_rate}% win rate")

        return "\n".join(lines) if lines else "- No opening data available"

    def _get_opponent_weak_openings(self) -> str:
        """Find opponent's weakest openings (as Black)."""
        opening_stats = self.opponent_stats.get("opening_stats", {})

        weak_openings = sorted(
            [(name, data) for name, data in opening_stats.items()
             if data.get("total_games", 0) >= 3],
            key=lambda x: x[1].get("win_rate", 100)
        )[:3]

        if not weak_openings:
            return "- No significant weak openings detected"

        return "\n".join([
            f"- {name}: {data['win_rate']}% win rate ({data['total_games']} games)"
            for name, data in weak_openings
        ])

    def _get_player_best_openings(self) -> str:
        """Find player's best openings by win rate."""
        opening_stats = self.player_stats.get("opening_stats", {})

        best_openings = sorted(
            opening_stats.items(),
            key=lambda x: (-x[1].get("win_rate", 0), -x[1].get("total_games", 0))
        )[:3]

        if not best_openings:
            return "- No data available"

        return "\n".join([
            f"- {name}: {data['win_rate']}% win rate ({data['total_games']} games)"
            for name, data in best_openings
        ])

    def render(self) -> str:
        """Render the battle plan prompt."""
        opponent_openings = self._format_openings_for_scouting(
            self.opponent_stats.get("opening_stats", {})
        )
        player_openings = self._format_openings_for_scouting(
            self.player_stats.get("opening_stats", {})
        )
        opponent_weak_str = self._get_opponent_weak_openings()
        player_best_str = self._get_player_best_openings()

        opp_white_wr = self._calc_color_win_rate(self.opponent_stats, "white")
        opp_black_wr = self._calc_color_win_rate(self.opponent_stats, "black")
        player_white_wr = self._calc_color_win_rate(self.player_stats, "white")
        player_black_wr = self._calc_color_win_rate(self.player_stats, "black")
        loss_methods = self.opponent_stats.get('loss_methods', {})

        return f"""You are a chess coach preparing a battle plan for {self.player_username} to BEAT {self.opponent_username}.
Focus entirely on how {self.player_username} can WIN this match using their strengths against the opponent's weaknesses.

OPPONENT ({self.opponent_username}) - THE TARGET:
- Record: {self.opponent_stats.get('wins', 0)}W / {self.opponent_stats.get('losses', 0)}L / {self.opponent_stats.get('draws', 0)}D ({self.opponent_stats.get('total_games', 0)} games)
- White win rate: {opp_white_wr}%
- Black win rate: {opp_black_wr}%
- Their openings:
{opponent_openings}
- Their WEAKEST openings (exploit these!):
{opponent_weak_str}
- How they lose: {loss_methods}

YOUR PLAYER ({self.player_username}) - STRENGTHS TO LEVERAGE:
- Record: {self.player_stats.get('wins', 0)}W / {self.player_stats.get('losses', 0)}L / {self.player_stats.get('draws', 0)}D ({self.player_stats.get('total_games', 0)} games)
- White win rate: {player_white_wr}%
- Black win rate: {player_black_wr}%
- Their best openings (highest win rate):
{player_best_str}
- Their most played openings:
{player_openings}

Provide a battle plan in JSON format (no markdown code blocks, just raw JSON):
{{
    "game_plan": "2-3 paragraphs of overall strategy. Start with the key insight about this matchup. What style should {self.player_username} play? Should they be aggressive or solid? How can they use their strengths against opponent weaknesses? Be specific.",
    "white_strategy": "1-2 paragraphs. What specific openings should {self.player_username} play as White? Consider BOTH their own best white openings AND the opponent's weak defenses. Recommend concrete opening choices.",
    "black_strategy": "1-2 paragraphs. What specific openings should {self.player_username} play as Black? Consider their own strongest black defenses AND what the opponent typically plays as White. Recommend concrete responses.",
    "weaknesses": "Bullet list of 3-5 specific opponent weaknesses to TARGET. Use format: - **Name:** Description. Be actionable.",
    "watch_out": "Bullet list of 2-3 opponent strengths to AVOID. Use format: - **Name:** Description. What traps should {self.player_username} not fall into?",
    "white_tips": "3-4 quick tips for the White game. Use format: - ☐ Tip text",
    "black_tips": "3-4 quick tips for the Black game. Use format: - ☐ Tip text",
    "white_opening_recommendation": {{
        "name": "Name of recommended opening (e.g., Italian Game, London System)",
        "first_moves": "e.g., 1. e4 e5 2. Nf3 Nc6 3. Bc4",
        "why": "1-2 sentences explaining why this opening exploits the opponent's weaknesses",
        "key_idea": "The main strategic idea to aim for (1 sentence)",
        "watch_for": "What the opponent might try that you should be ready for"
    }},
    "black_opening_recommendation": {{
        "name": "Name of recommended defense (e.g., Sicilian Defense, King's Indian)",
        "against": "What White opening this responds to (e.g., 1. e4, 1. d4)",
        "first_moves": "e.g., 1. e4 c5 2. Nf3 d6",
        "why": "1-2 sentences explaining why this defense works against this opponent",
        "key_idea": "The main counterplay idea (1 sentence)",
        "watch_for": "Traps or tactics the opponent might try"
    }}
}}

Be aggressive and actionable. This is a battle plan to WIN, not a neutral analysis.

IMPORTANT: On Chess.com and Lichess, players do NOT get to choose their color - it's assigned randomly. Never suggest "choosing" to play White or Black. Instead, provide preparation for BOTH colors since either could happen."""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return prompt metadata."""
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "recommended_tier": self.recommended_tier.value,
            "opponent": self.opponent_username,
            "player": self.player_username,
            "opponent_games": self.opponent_stats.get("total_games", 0),
            "player_games": self.player_stats.get("total_games", 0),
        }
