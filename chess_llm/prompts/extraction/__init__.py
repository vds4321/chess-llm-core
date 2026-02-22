"""
Extraction prompt templates -- structured data parsing from free-text reports.

Templates:
    - ``KeyAreasExtractionPrompt`` -- extracts improvement areas as JSON.
"""

from chess_llm.prompts.extraction.key_areas import KeyAreasExtractionPrompt

__all__ = [
    "KeyAreasExtractionPrompt",
]
