"""
Base prompt template framework.

All prompt templates in the library inherit from ``PromptTemplate``.
This module provides:

    - ``PromptVersion`` -- semantic versioning for A/B testing and auditing
    - ``OutputFormat``  -- enum declaring the expected LLM response format
    - ``PromptTemplate`` -- abstract base with rendering, validation, and
      response parsing hooks

Design rationale:
    Prompt templates encode domain knowledge (chess coaching) and decouple
    *what* we ask the LLM from *how* we communicate with it.  Each
    template declares its recommended model tier and token budget so the
    calling code can select the right provider automatically.

Creating a new prompt:
    1. Subclass ``PromptTemplate`` as a ``@dataclass``.
    2. Override class-level attributes (``prompt_id``, ``version``, etc.).
    3. Implement ``render()`` returning the full prompt text.
    4. Optionally override ``render_system_prompt()``, ``validate()``,
       and ``parse_response()``.
    5. Re-export from the appropriate ``__init__.py``.

Maintenance:
    When changing a prompt's text, bump its ``PromptVersion``.  This
    enables downstream analytics in ``kaspar_eval`` to distinguish
    results produced by different prompt revisions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from chess_llm.config.models import ModelTier


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


@dataclass
class PromptVersion:
    """
    Semantic version for a prompt template.

    Follows ``major.minor.patch`` convention:
        - *major*: Breaking changes to output format or required fields.
        - *minor*: Meaningful prompt text changes that may affect quality.
        - *patch*: Cosmetic fixes (typos, formatting) with no quality impact.
    """

    major: int = 1
    minor: int = 0
    patch: int = 0
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def version_string(self) -> str:
        """Return version as ``"major.minor.patch"``."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        return self.version_string


# ---------------------------------------------------------------------------
# Output format enum
# ---------------------------------------------------------------------------


class OutputFormat(Enum):
    """Declares what the LLM is expected to return."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    JSON_ARRAY = "json_array"
    STRUCTURED = "structured"


# ---------------------------------------------------------------------------
# Abstract base template
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate(ABC):
    """
    Abstract base for all prompt templates.

    Subclasses define:
        - Template text with data placeholders
        - Required and optional input parameters
        - Recommended model tier and token budget
        - Output format expectation
        - Optional response-parsing logic

    Example::

        @dataclass
        class MyPrompt(PromptTemplate):
            prompt_id = "my_prompt"
            version = PromptVersion(1, 0, 0)
            recommended_tier = ModelTier.STANDARD

            name: str = ""

            def render(self) -> str:
                return f"Hello, {self.name}!"
    """

    # -- Override these in subclasses ---------------------------------------
    prompt_id: str = "base"
    version: PromptVersion = PromptVersion(1, 0, 0)
    recommended_tier: ModelTier = ModelTier.STANDARD
    output_format: OutputFormat = OutputFormat.MARKDOWN
    estimated_input_tokens: int = 500
    estimated_output_tokens: int = 1000

    # -----------------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------------

    @abstractmethod
    def render(self) -> str:
        """
        Render the complete prompt text, ready to send to the LLM.

        Returns:
            The fully-interpolated prompt string.
        """
        ...

    def render_system_prompt(self) -> Optional[str]:
        """
        Optional system prompt that sets the LLM's persona / context.

        Returns:
            System prompt text, or ``None`` to omit.
        """
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return a serialisable dict describing this prompt template.

        Useful for logging, analytics, and linking results back to the
        exact prompt version that produced them.
        """
        return {
            "prompt_id": self.prompt_id,
            "version": self.version.version_string,
            "recommended_tier": self.recommended_tier.value,
            "output_format": self.output_format.value,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
        }

    def validate(self) -> List[str]:
        """
        Validate prompt parameters before rendering.

        Returns:
            A list of human-readable error messages.  Empty means valid.
        """
        return []

    def parse_response(self, response: str) -> Any:
        """
        Parse the raw LLM response into structured data.

        The default implementation handles JSON extraction from markdown
        code blocks.  Override for custom formats.

        Args:
            response: Raw text returned by the LLM.

        Returns:
            Parsed data (dict, list, or raw string depending on format).
        """
        if self.output_format in (OutputFormat.JSON, OutputFormat.JSON_ARRAY):
            return _extract_json(response)
        return response


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Any:
    """
    Extract and parse JSON from an LLM response.

    Handles common patterns:
        - Raw JSON
        - JSON wrapped in markdown ````` fences
        - ``json`` language tag after the opening fence
    """
    import json

    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (possibly with language tag)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Remove stray "json" prefix
    if text.startswith("json"):
        text = text[4:].strip()

    return json.loads(text)


def create_prompt_from_template(template: str, **kwargs: Any) -> str:
    """
    Render a simple Python format-string template.

    This is a lightweight alternative for ad-hoc prompts that don't need
    the full ``PromptTemplate`` machinery.

    Args:
        template: Format string with ``{placeholder}`` markers.
        **kwargs: Values to substitute.

    Returns:
        The rendered string.
    """
    return template.format(**kwargs)
