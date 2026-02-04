"""
Base LLM provider protocol and types.

This module defines the abstract interface that all LLM providers must implement.
Using Protocol (PEP 544) enables structural subtyping - implementations don't need
to explicitly inherit, they just need to implement the required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, provider: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class RateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when API authentication fails."""

    pass


class TokenLimitError(ProviderError):
    """Raised when token limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        requested_tokens: int = 0,
        max_tokens: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)
        self.requested_tokens = requested_tokens
        self.max_tokens = max_tokens


@dataclass
class LLMConfig:
    """Configuration for an LLM request."""

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None

    # Provider-specific options
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    """Token usage statistics from a response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost tracking (in USD)
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    @property
    def cost_formatted(self) -> str:
        """Return cost as formatted string."""
        return f"${self.total_cost:.6f}"


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    provider: str

    # Usage statistics
    usage: UsageStats = field(default_factory=UsageStats)

    # Metadata
    finish_reason: str = "stop"
    created_at: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0

    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.content


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for LLM providers.

    All LLM providers (Anthropic, OpenAI, local models) must implement this interface.
    This enables consistent usage across different backends while allowing
    provider-specific optimizations.
    """

    @property
    def provider_id(self) -> str:
        """Unique identifier for this provider (e.g., 'anthropic', 'openai')."""
        ...

    @property
    def model_id(self) -> str:
        """Current model identifier (e.g., 'claude-3-5-sonnet-20241022')."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for display (e.g., 'Claude 3.5 Sonnet')."""
        ...

    @property
    def supports_vision(self) -> bool:
        """Whether this provider/model supports image inputs."""
        ...

    @property
    def max_context_tokens(self) -> int:
        """Maximum context window size in tokens."""
        ...

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens supported."""
        ...

    def complete(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt/message
            config: Optional configuration (uses defaults if not provided)

        Returns:
            LLMResponse with the generated content and metadata

        Raises:
            ProviderError: For general provider errors
            RateLimitError: When rate limited
            AuthenticationError: When authentication fails
            TokenLimitError: When token limit exceeded
        """
        ...

    def complete_with_images(
        self,
        prompt: str,
        images: List[bytes],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion with image context.

        Args:
            prompt: The user prompt/message
            images: List of images as bytes (PNG/JPEG)
            config: Optional configuration

        Returns:
            LLMResponse with the generated content

        Raises:
            NotImplementedError: If provider doesn't support vision
            ProviderError: For general provider errors
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.

        This is an approximation - actual tokenization may differ.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        ...

    def get_cost_per_token(self) -> tuple[float, float]:
        """
        Get the cost per token for this model.

        Returns:
            Tuple of (input_cost_per_million, output_cost_per_million) in USD
        """
        ...


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides common functionality and enforces the LLMProvider protocol.
    Concrete implementations should inherit from this class.
    """

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the provider.

        Args:
            model_id: The model identifier to use
            api_key: API key (if not provided, reads from environment)
            **kwargs: Provider-specific options
        """
        self._model_id = model_id
        self._api_key = api_key
        self._options = kwargs

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        ...

    @property
    def model_id(self) -> str:
        """Current model identifier."""
        return self._model_id

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""
        ...

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this model supports images."""
        ...

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum context window."""
        ...

    @property
    @abstractmethod
    def max_output_tokens(self) -> int:
        """Maximum output tokens."""
        ...

    @abstractmethod
    def complete(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion."""
        ...

    def complete_with_images(
        self,
        prompt: str,
        images: List[bytes],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion with images."""
        if not self.supports_vision:
            raise NotImplementedError(
                f"Provider {self.provider_id} model {self.model_id} does not support vision"
            )
        # Subclasses that support vision should override this
        raise NotImplementedError("Vision support not implemented")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens using a simple heuristic.

        Override this method for provider-specific tokenization.
        """
        # Rough estimate: ~4 characters per token for English text
        return len(text) // 4

    @abstractmethod
    def get_cost_per_token(self) -> tuple[float, float]:
        """Get cost per million tokens (input, output)."""
        ...

    def _calculate_cost(self, usage: UsageStats) -> UsageStats:
        """Calculate cost based on usage and pricing."""
        input_price, output_price = self.get_cost_per_token()
        usage.input_cost = (usage.input_tokens / 1_000_000) * input_price
        usage.output_cost = (usage.output_tokens / 1_000_000) * output_price
        usage.total_cost = usage.input_cost + usage.output_cost
        return usage

    def _get_default_config(self) -> LLMConfig:
        """Return default configuration."""
        return LLMConfig()
