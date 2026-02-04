"""Usage tracking for LLM requests."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class UsageRecord:
    """Record of a single LLM request."""

    timestamp: datetime
    provider: str
    model: str
    prompt_id: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens generated per second."""
        if self.latency_ms <= 0:
            return 0.0
        return self.output_tokens / (self.latency_ms / 1000)


@dataclass
class UsageTracker:
    """
    Track LLM usage across requests.

    Thread-safe for basic operations. For high-concurrency applications,
    consider using a dedicated metrics system.
    """

    records: List[UsageRecord] = field(default_factory=list)
    budget_limit_usd: Optional[float] = None

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        prompt_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> UsageRecord:
        """
        Record a single LLM request.

        Args:
            provider: Provider name (e.g., 'anthropic')
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            latency_ms: Request latency in milliseconds
            prompt_id: Optional prompt template identifier
            success: Whether the request succeeded
            error: Error message if failed

        Returns:
            The created UsageRecord
        """
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_id=prompt_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        self.records.append(record)
        return record

    @property
    def total_cost(self) -> float:
        """Get total cost across all records."""
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Get total tokens across all records."""
        return sum(r.total_tokens for r in self.records)

    @property
    def total_requests(self) -> int:
        """Get total number of requests."""
        return len(self.records)

    @property
    def successful_requests(self) -> int:
        """Get number of successful requests."""
        return sum(1 for r in self.records if r.success)

    @property
    def failed_requests(self) -> int:
        """Get number of failed requests."""
        return sum(1 for r in self.records if not r.success)

    @property
    def average_latency_ms(self) -> float:
        """Get average latency across all requests."""
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    @property
    def budget_remaining(self) -> Optional[float]:
        """Get remaining budget, if a limit is set."""
        if self.budget_limit_usd is None:
            return None
        return max(0.0, self.budget_limit_usd - self.total_cost)

    @property
    def is_over_budget(self) -> bool:
        """Check if over budget."""
        if self.budget_limit_usd is None:
            return False
        return self.total_cost >= self.budget_limit_usd

    def get_cost_by_provider(self) -> Dict[str, float]:
        """Get costs grouped by provider."""
        costs: Dict[str, float] = {}
        for record in self.records:
            costs[record.provider] = costs.get(record.provider, 0.0) + record.cost_usd
        return costs

    def get_cost_by_model(self) -> Dict[str, float]:
        """Get costs grouped by model."""
        costs: Dict[str, float] = {}
        for record in self.records:
            costs[record.model] = costs.get(record.model, 0.0) + record.cost_usd
        return costs

    def get_cost_by_prompt(self) -> Dict[str, float]:
        """Get costs grouped by prompt template."""
        costs: Dict[str, float] = {}
        for record in self.records:
            key = record.prompt_id or "unknown"
            costs[key] = costs.get(key, 0.0) + record.cost_usd
        return costs

    def get_summary(self) -> Dict[str, any]:
        """Get a summary of all usage."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "cost_by_provider": self.get_cost_by_provider(),
            "cost_by_model": self.get_cost_by_model(),
            "cost_by_prompt": self.get_cost_by_prompt(),
            "budget_limit_usd": self.budget_limit_usd,
            "budget_remaining_usd": self.budget_remaining,
            "is_over_budget": self.is_over_budget,
        }

    def clear(self) -> None:
        """Clear all records."""
        self.records.clear()


# Global tracker instance
_global_tracker: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    """Get the global usage tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = UsageTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset the global usage tracker."""
    global _global_tracker
    _global_tracker = UsageTracker()
