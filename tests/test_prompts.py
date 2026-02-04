"""Tests for prompt template modules."""

import pytest

from chess_llm.config.models import ModelTier
from chess_llm.prompts.base import PromptTemplate, PromptVersion
from chess_llm.prompts.coaching.mentor_insights import MentorInsightsPrompt
from chess_llm.prompts.extraction.key_areas import KeyAreasExtractionPrompt
from chess_llm.prompts.scouting.battle_plan import BattlePlanPrompt


class TestPromptVersion:
    """Test PromptVersion dataclass."""

    def test_version_creation(self):
        """Test creating a prompt version."""
        version = PromptVersion(
            major=1,
            minor=0,
            patch=0,
            description="Test version",
        )
        assert version.major == 1
        assert version.version_string == "1.0.0"


class TestMentorInsightsPrompt:
    """Test MentorInsightsPrompt."""

    @pytest.fixture
    def sample_stats(self):
        """Sample player statistics."""
        return {
            "total_games": 100,
            "wins": 55,
            "losses": 35,
            "draws": 10,
            "white_wins": 30,
            "white_games": 50,
            "black_wins": 25,
            "black_games": 50,
            "opening_stats": {
                "Italian Game": {
                    "total_games": 20,
                    "wins": 12,
                    "win_rate": 60.0,
                },
                "Sicilian Defense": {
                    "total_games": 15,
                    "wins": 8,
                    "win_rate": 53.3,
                },
            },
            "time_of_day_performance": {
                "morning": {"games": 30, "win_rate": 60},
                "afternoon": {"games": 40, "win_rate": 50},
                "evening": {"games": 30, "win_rate": 55},
            },
        }

    @pytest.fixture
    def sample_progression(self):
        """Sample progression data."""
        return {
            "rating_change": 50,
            "rating_start": 1200,
            "rating_end": 1250,
            "win_rate_trend": "improving",
            "recent_games": 20,
        }

    def test_create_prompt(self, sample_stats, sample_progression):
        """Test creating the prompt."""
        prompt = MentorInsightsPrompt(
            username="TestPlayer",
            stats=sample_stats,
            progression=sample_progression,
        )
        assert prompt.username == "TestPlayer"
        assert prompt.prompt_id == "mentor_insights"
        assert prompt.recommended_tier == ModelTier.STANDARD

    def test_render_prompt(self, sample_stats, sample_progression):
        """Test rendering the prompt."""
        prompt = MentorInsightsPrompt(
            username="TestPlayer",
            stats=sample_stats,
            progression=sample_progression,
        )
        rendered = prompt.render()

        assert "TestPlayer" in rendered
        assert "100" in rendered  # total games
        assert isinstance(rendered, str)
        assert len(rendered) > 100

    def test_get_metadata(self, sample_stats, sample_progression):
        """Test prompt metadata."""
        prompt = MentorInsightsPrompt(
            username="TestPlayer",
            stats=sample_stats,
            progression=sample_progression,
        )
        metadata = prompt.get_metadata()

        assert metadata["prompt_id"] == "mentor_insights"
        assert "version" in metadata
        assert "recommended_tier" in metadata


class TestKeyAreasExtractionPrompt:
    """Test KeyAreasExtractionPrompt."""

    def test_create_prompt(self):
        """Test creating extraction prompt."""
        prompt = KeyAreasExtractionPrompt(
            report_content="This is a sample report about improving openings..."
        )
        assert prompt.prompt_id == "key_areas_extraction"
        assert prompt.recommended_tier == ModelTier.CHEAP

    def test_render_prompt(self):
        """Test rendering extraction prompt."""
        report = """
        # Coaching Report

        ## Opening Analysis
        Your Sicilian Defense needs work. Focus on the Najdorf variation.

        ## Tactical Training
        Practice knight forks and discovered attacks.

        ## Endgame Study
        Work on rook endgames, especially Lucena position.
        """
        prompt = KeyAreasExtractionPrompt(report_content=report)
        rendered = prompt.render()

        assert "Sicilian Defense" in rendered or "report" in rendered.lower()
        assert "JSON" in rendered  # Should ask for JSON output


class TestBattlePlanPrompt:
    """Test BattlePlanPrompt."""

    @pytest.fixture
    def opponent_stats(self):
        """Sample opponent statistics."""
        return {
            "total_games": 150,
            "wins": 75,
            "losses": 60,
            "draws": 15,
            "white_wins": 45,
            "white_games": 75,
            "black_wins": 30,
            "black_games": 75,
            "opening_stats": {
                "London System": {
                    "total_games": 40,
                    "wins": 24,
                    "win_rate": 60.0,
                },
                "Caro-Kann Defense": {
                    "total_games": 25,
                    "wins": 10,
                    "win_rate": 40.0,
                },
            },
            "loss_methods": {
                "timeout": 10,
                "resigned": 35,
                "checkmated": 15,
            },
        }

    @pytest.fixture
    def player_stats(self):
        """Sample player statistics."""
        return {
            "total_games": 200,
            "wins": 110,
            "losses": 70,
            "draws": 20,
            "white_wins": 60,
            "white_games": 100,
            "black_wins": 50,
            "black_games": 100,
            "opening_stats": {
                "Italian Game": {
                    "total_games": 50,
                    "wins": 32,
                    "win_rate": 64.0,
                },
                "Sicilian Defense": {
                    "total_games": 35,
                    "wins": 20,
                    "win_rate": 57.1,
                },
            },
        }

    def test_create_prompt(self, opponent_stats, player_stats):
        """Test creating battle plan prompt."""
        prompt = BattlePlanPrompt(
            opponent_username="Opponent123",
            player_username="Player456",
            opponent_stats=opponent_stats,
            player_stats=player_stats,
        )
        assert prompt.prompt_id == "battle_plan"
        assert prompt.recommended_tier == ModelTier.STANDARD

    def test_render_prompt(self, opponent_stats, player_stats):
        """Test rendering battle plan prompt."""
        prompt = BattlePlanPrompt(
            opponent_username="Opponent123",
            player_username="Player456",
            opponent_stats=opponent_stats,
            player_stats=player_stats,
        )
        rendered = prompt.render()

        assert "Opponent123" in rendered
        assert "Player456" in rendered
        assert "BEAT" in rendered
        assert "battle plan" in rendered.lower()
        assert "JSON" in rendered  # Should ask for JSON output

    def test_metadata(self, opponent_stats, player_stats):
        """Test battle plan metadata."""
        prompt = BattlePlanPrompt(
            opponent_username="Opponent123",
            player_username="Player456",
            opponent_stats=opponent_stats,
            player_stats=player_stats,
        )
        metadata = prompt.metadata

        assert metadata["prompt_id"] == "battle_plan"
        assert metadata["opponent"] == "Opponent123"
        assert metadata["player"] == "Player456"
        assert metadata["opponent_games"] == 150
        assert metadata["player_games"] == 200

    def test_color_win_rate_calculation(self, opponent_stats, player_stats):
        """Test color win rate calculation."""
        prompt = BattlePlanPrompt(
            opponent_username="Opponent123",
            player_username="Player456",
            opponent_stats=opponent_stats,
            player_stats=player_stats,
        )

        # Opponent: 45/75 = 60% white win rate
        assert prompt._calc_color_win_rate(opponent_stats, "white") == 60.0
        # Opponent: 30/75 = 40% black win rate
        assert prompt._calc_color_win_rate(opponent_stats, "black") == 40.0

    def test_handles_empty_stats(self):
        """Test handling empty statistics."""
        prompt = BattlePlanPrompt(
            opponent_username="Opponent",
            player_username="Player",
            opponent_stats={},
            player_stats={},
        )
        rendered = prompt.render()

        # Should not crash and should contain usernames
        assert "Opponent" in rendered
        assert "Player" in rendered
