"""
Usage and cost tracking sub-package.

Provides per-request recording, aggregate metrics, and budget enforcement
for all LLM calls made through the library.

Exports:
    - ``UsageTracker`` -- the core tracker class.
    - ``UsageRecord``  -- a single request record.
    - ``get_tracker()`` / ``reset_tracker()`` -- global singleton management.
"""

from chess_llm.tracking.usage import (
    UsageTracker,
    UsageRecord,
    get_tracker,
    reset_tracker,
)

__all__ = [
    "UsageTracker",
    "UsageRecord",
    "get_tracker",
    "reset_tracker",
]
