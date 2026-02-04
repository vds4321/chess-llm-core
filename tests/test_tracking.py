"""Tests for tracking modules."""

import pytest
from datetime import datetime

from chess_llm.tracking.usage import (
    UsageRecord,
    UsageTracker,
    get_tracker,
    reset_tracker,
)


class TestUsageRecord:
    """Test UsageRecord dataclass."""

    def test_create_record(self):
        """Test creating a usage record."""
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            prompt_id="test_prompt",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost_usd=0.0005,
            latency_ms=150.0,
            success=True,
        )
        assert record.provider == "anthropic"
        assert record.total_tokens == 300
        assert record.success is True

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="test",
            prompt_id=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost_usd=0.001,
            latency_ms=1000.0,  # 1 second
            success=True,
        )
        # 200 output tokens / 1 second = 200 tokens/sec
        assert record.tokens_per_second == 200.0

    def test_tokens_per_second_zero_latency(self):
        """Test tokens per second with zero latency."""
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="test",
            prompt_id=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost_usd=0.001,
            latency_ms=0.0,
            success=True,
        )
        assert record.tokens_per_second == 0.0

    def test_failed_record(self):
        """Test creating a failed record."""
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="test",
            prompt_id=None,
            input_tokens=100,
            output_tokens=0,
            total_tokens=100,
            cost_usd=0.0001,
            latency_ms=50.0,
            success=False,
            error="Rate limit exceeded",
        )
        assert record.success is False
        assert record.error == "Rate limit exceeded"


class TestUsageTracker:
    """Test UsageTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker."""
        return UsageTracker()

    def test_empty_tracker(self, tracker):
        """Test empty tracker stats."""
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert tracker.total_requests == 0
        assert tracker.average_latency_ms == 0.0

    def test_record_usage(self, tracker):
        """Test recording usage."""
        record = tracker.record(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.0005,
            latency_ms=150.0,
            prompt_id="test_prompt",
        )
        assert isinstance(record, UsageRecord)
        assert tracker.total_requests == 1
        assert tracker.total_tokens == 300
        assert tracker.total_cost == 0.0005

    def test_multiple_records(self, tracker):
        """Test multiple records."""
        tracker.record(
            provider="anthropic",
            model="haiku",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
        )
        tracker.record(
            provider="anthropic",
            model="sonnet",
            input_tokens=200,
            output_tokens=400,
            cost_usd=0.003,
            latency_ms=200.0,
        )
        assert tracker.total_requests == 2
        assert tracker.total_tokens == 900
        assert tracker.total_cost == 0.004
        assert tracker.average_latency_ms == 150.0

    def test_successful_and_failed_counts(self, tracker):
        """Test counting successful and failed requests."""
        tracker.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
            success=True,
        )
        tracker.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=0,
            cost_usd=0.0001,
            latency_ms=50.0,
            success=False,
            error="Rate limited",
        )
        assert tracker.successful_requests == 1
        assert tracker.failed_requests == 1

    def test_cost_by_provider(self, tracker):
        """Test cost grouped by provider."""
        tracker.record(
            provider="anthropic",
            model="haiku",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
        )
        tracker.record(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.002,
            latency_ms=150.0,
        )
        costs = tracker.get_cost_by_provider()
        assert costs["anthropic"] == 0.001
        assert costs["openai"] == 0.002

    def test_cost_by_model(self, tracker):
        """Test cost grouped by model."""
        tracker.record(
            provider="anthropic",
            model="haiku",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
        )
        tracker.record(
            provider="anthropic",
            model="sonnet",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.003,
            latency_ms=150.0,
        )
        costs = tracker.get_cost_by_model()
        assert costs["haiku"] == 0.001
        assert costs["sonnet"] == 0.003

    def test_cost_by_prompt(self, tracker):
        """Test cost grouped by prompt template."""
        tracker.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
            prompt_id="mentor_insights",
        )
        tracker.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.0005,
            latency_ms=100.0,
            prompt_id="key_areas_extraction",
        )
        costs = tracker.get_cost_by_prompt()
        assert costs["mentor_insights"] == 0.001
        assert costs["key_areas_extraction"] == 0.0005

    def test_budget_tracking(self):
        """Test budget limit tracking."""
        tracker = UsageTracker(budget_limit_usd=0.01)

        assert tracker.budget_remaining == 0.01
        assert tracker.is_over_budget is False

        tracker.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.005,
            latency_ms=100.0,
        )
        assert tracker.budget_remaining == 0.005
        assert tracker.is_over_budget is False

        tracker.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.006,
            latency_ms=100.0,
        )
        assert tracker.budget_remaining == 0.0
        assert tracker.is_over_budget is True

    def test_no_budget_limit(self, tracker):
        """Test when no budget limit is set."""
        assert tracker.budget_remaining is None
        assert tracker.is_over_budget is False

    def test_get_summary(self, tracker):
        """Test getting usage summary."""
        tracker.record(
            provider="anthropic",
            model="haiku",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
            prompt_id="test",
        )
        summary = tracker.get_summary()

        assert summary["total_requests"] == 1
        assert summary["total_tokens"] == 300
        assert summary["total_cost_usd"] == 0.001
        assert "cost_by_provider" in summary
        assert "cost_by_model" in summary
        assert "cost_by_prompt" in summary

    def test_clear(self, tracker):
        """Test clearing tracker."""
        tracker.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
        )
        assert tracker.total_requests == 1

        tracker.clear()
        assert tracker.total_requests == 0
        assert tracker.total_cost == 0.0


class TestGlobalTracker:
    """Test global tracker functions."""

    def test_get_tracker(self):
        """Test getting global tracker."""
        tracker = get_tracker()
        assert isinstance(tracker, UsageTracker)

    def test_get_tracker_singleton(self):
        """Test tracker is singleton."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()
        assert tracker1 is tracker2

    def test_reset_tracker(self):
        """Test resetting global tracker."""
        tracker1 = get_tracker()
        tracker1.record(
            provider="anthropic",
            model="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            latency_ms=100.0,
        )
        reset_tracker()
        tracker2 = get_tracker()

        # Should be a new tracker with no records
        assert tracker2.total_requests == 0
