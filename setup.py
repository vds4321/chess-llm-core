"""
Fallback setup.py for older pip versions that don't support pyproject.toml.

The canonical build configuration is in pyproject.toml.  This file exists
solely for backwards compatibility with pip < 21.3.
"""
from setuptools import setup

setup(
    name="chess-llm-core",
    version="0.2.0",
    packages=[
        "chess_llm",
        "chess_llm.config",
        "chess_llm.providers",
        "chess_llm.prompts",
        "chess_llm.prompts.coaching",
        "chess_llm.prompts.scouting",
        "chess_llm.prompts.extraction",
        "chess_llm.tracking",
    ],
    install_requires=["pydantic>=2.0.0"],
    extras_require={
        "anthropic": ["anthropic>=0.18.0"],
        "openai": ["openai>=1.0.0"],
        "all": ["anthropic>=0.18.0", "openai>=1.0.0"],
    },
    python_requires=">=3.9",
)
