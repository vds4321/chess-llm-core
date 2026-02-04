"""
Base prompt template classes.

Provides:
- PromptTemplate: Base class for all prompt templates
- PromptVersion: Version tracking for prompts
- Output parsing utilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from chess_llm.config.models import ModelTier


@dataclass
class PromptVersion:
    """Version information for a prompt template."""

    major: int = 1
    minor: int = 0
    patch: int = 0
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def version_string(self) -> str:
        """Return version as string (e.g., '1.2.3')."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        return self.version_string


class OutputFormat(Enum):
    """Expected output format from the LLM."""

    TEXT = "text"  # Free-form text
    MARKDOWN = "markdown"  # Markdown formatted text
    JSON = "json"  # JSON object
    JSON_ARRAY = "json_array"  # JSON array
    STRUCTURED = "structured"  # Custom structured format


@dataclass
class PromptTemplate(ABC):
    """
    Base class for prompt templates.

    Subclasses define specific prompts with:
    - Template text with placeholders
    - Required and optional parameters
    - Recommended model tier and token limits
    - Output format expectations

    Example:
        class MyPrompt(PromptTemplate):
            prompt_id = "my_prompt"
            version = PromptVersion(1, 0, 0)
            recommended_tier = ModelTier.STANDARD

            def __init__(self, name: str):
                self.name = name

            def render(self) -> str:
                return f"Hello, {self.name}!"
    """

    # Class-level attributes (override in subclasses)
    prompt_id: str = "base"
    version: PromptVersion = PromptVersion(1, 0, 0)
    recommended_tier: ModelTier = ModelTier.STANDARD
    output_format: OutputFormat = OutputFormat.MARKDOWN
    estimated_input_tokens: int = 500
    estimated_output_tokens: int = 1000

    @abstractmethod
    def render(self) -> str:
        """
        Render the prompt template with current parameters.

        Returns:
            The complete prompt text ready for the LLM
        """
        ...

    def render_system_prompt(self) -> Optional[str]:
        """
        Render the system prompt if applicable.

        Override this method to provide a system prompt.

        Returns:
            System prompt text, or None if not needed
        """
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this prompt.

        Returns:
            Dict with prompt ID, version, tier, etc.
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
        Validate the prompt parameters.

        Override to add custom validation.

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def parse_response(self, response: str) -> Any:
        """
        Parse the LLM response into structured data.

        Default implementation returns raw text. Override for structured outputs.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed response (varies by output_format)
        """
        if self.output_format == OutputFormat.JSON:
            import json
            # Try to extract JSON from response
            text = response.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            if text.startswith("json"):
                text = text[4:].strip()
            return json.loads(text)

        elif self.output_format == OutputFormat.JSON_ARRAY:
            import json
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return json.loads(text)

        else:
            return response


def create_prompt_from_template(
    template: str,
    **kwargs: Any,
) -> str:
    """
    Simple template rendering using Python format strings.

    Args:
        template: Template string with {placeholder} markers
        **kwargs: Values to substitute

    Returns:
        Rendered template string
    """
    return template.format(**kwargs)
