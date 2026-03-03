"""
Retry logic for LLM provider calls.

Provides exponential backoff with jitter for transient errors
(rate limits, server overload). Applied automatically by
``BaseLLMProvider.acomplete()`` and ``BaseLLMProvider.acomplete_messages()``.

Usage::

    from chess_llm.retry import with_retry

    result = await with_retry(
        lambda: provider.acomplete(prompt, config),
        max_retries=3,
    )

The retry decorator respects ``RateLimitError.retry_after`` when
the provider supplies a suggested delay.
"""

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

from chess_llm.providers.base import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    TokenLimitError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errors that should NOT be retried (permanent failures)
_NON_RETRYABLE = (AuthenticationError, TokenLimitError)


async def with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
) -> T:
    """
    Execute an async callable with exponential backoff on transient errors.

    Only retries on ``RateLimitError`` and generic ``ProviderError``.
    Does NOT retry ``AuthenticationError`` or ``TokenLimitError`` (permanent).

    Args:
        coro_factory:   Callable that returns a new awaitable each time.
        max_retries:    Maximum number of retry attempts (0 = no retries).
        initial_delay:  Base delay in seconds before first retry.
        backoff_factor: Multiplier applied to delay after each retry.
        max_delay:      Maximum delay between retries.

    Returns:
        The result of the awaitable.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_error = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except _NON_RETRYABLE:
            # Permanent errors — don't retry
            raise
        except (RateLimitError, ProviderError) as e:
            last_error = e

            if attempt >= max_retries:
                # Exhausted all retries
                break

            # Use provider-suggested delay if available
            if isinstance(e, RateLimitError) and e.retry_after:
                wait = min(e.retry_after, max_delay)
            else:
                # Exponential backoff with jitter
                wait = min(delay * (1 + random.random() * 0.1), max_delay)
                delay *= backoff_factor

            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {wait:.1f}s: {e}"
            )
            await asyncio.sleep(wait)

    # All retries exhausted
    raise last_error
