# chess-llm-core

[![KasparChess](https://img.shields.io/badge/KasparChess-Live-blue)](https://kasparchess.com)

Shared LLM abstraction layer for chess coaching applications.

This library provides a provider-agnostic interface for working with large language models, purpose-built for chess coaching use cases.  It is the shared foundation for all Kaspar chess products.

## Products Using This Library

- [KasparChess](https://github.com/vds4321/KasparChess) - Parent-focused coaching for children (live at [kasparchess.com](https://kasparchess.com))
- [YourChessDotComCoach](https://github.com/vds4321/YourChessDotComCoach) - Adult chess coaching (live at yourchessdotcomcoach.fly.dev)

## Features

- **Protocol-based LLM provider abstraction** - flexible interface supporting multiple backends via PEP 544 structural subtyping
- **Model tier system** - CHEAP / STANDARD / PREMIUM tiers for automatic cost-quality routing
- **Versioned prompt templates** - reusable, testable prompts for coaching, scouting, and data extraction
- **Cost and usage tracking** - per-request and aggregate USD cost monitoring with budget limits
- **Provider registry** - factory-based provider management with lazy dependency loading
- **Security by default** - API keys are never logged, serialised, or exposed in `__repr__` output

## Architecture

```
chess_llm/
├── config/         # Model tiers, pricing, environment settings
│   ├── models.py   # ModelTier enum, ModelConfig, model registries
│   └── settings.py # ProviderSettings, Settings, env var loading
├── providers/      # LLM provider implementations
│   ├── base.py     # LLMProvider protocol, BaseLLMProvider ABC, response types
│   ├── anthropic.py# Anthropic Claude implementation (production-ready)
│   └── registry.py # Factory-based provider registration and lookup
├── prompts/        # Versioned prompt templates
│   ├── base.py     # PromptTemplate ABC, PromptVersion, output parsing
│   ├── coaching/   # MentorInsightsPrompt, OpeningAnalysisPrompt
│   ├── extraction/ # KeyAreasExtractionPrompt (JSON output)
│   └── scouting/   # BattlePlanPrompt (opponent analysis)
└── tracking/       # Usage and cost tracking
    └── usage.py    # UsageTracker, UsageRecord, budget enforcement
```

## Installation

```bash
# Basic installation (no providers)
pip install git+https://github.com/vds4321/chess-llm-core.git

# With Anthropic support
pip install "git+https://github.com/vds4321/chess-llm-core.git#egg=chess-llm-core[anthropic]"

# With all providers
pip install "git+https://github.com/vds4321/chess-llm-core.git#egg=chess-llm-core[all]"

# Development (includes test and lint tools)
pip install -e ".[dev]"
```

## Quick Start

```python
from chess_llm import get_provider, ModelTier
from chess_llm.prompts import MentorInsightsPrompt

# Get a provider for a specific tier
provider = get_provider("anthropic", tier=ModelTier.STANDARD)

# Use a prompt template
prompt = MentorInsightsPrompt(
    username="player123",
    stats=player_stats,
    progression=progression_data,
)

# Generate response
response = provider.complete(prompt.render())
print(response.content)
```

## Model Tiers

| Tier       | Use Case                    | Anthropic Model    | Cost (input/output per 1M tokens) |
|------------|-----------------------------|--------------------|-----------------------------------|
| `CHEAP`    | Extraction, classification  | Claude 3.5 Haiku   | $1.00 / $5.00                     |
| `STANDARD` | Coaching, scouting, analysis| Claude Sonnet 4    | $3.00 / $15.00                    |
| `PREMIUM`  | Comprehensive reports       | Claude 3 Opus      | $15.00 / $75.00                   |

## Prompt Templates

### Coaching
- **`MentorInsightsPrompt`** - Personalised coaching report based on 6 months of game analysis.  Produces markdown with consistent section headers.
- **`OpeningAnalysisPrompt`** - Targeted advice for a specific opening (150-200 words).

### Scouting
- **`BattlePlanPrompt`** - Structured JSON battle plan for beating a specific opponent, with concrete opening recommendations.

### Extraction
- **`KeyAreasExtractionPrompt`** - Extracts structured JSON from free-text coaching reports for progress tracking.

## Configuration

Set these environment variables to configure the library:

| Variable                     | Description                      | Default      |
|------------------------------|----------------------------------|--------------|
| `ANTHROPIC_API_KEY`          | Anthropic API key                | *required*   |
| `OPENAI_API_KEY`             | OpenAI API key (future use)      | —            |
| `CHESS_LLM_DEFAULT_PROVIDER` | Default provider                 | `anthropic`  |
| `CHESS_LLM_DEFAULT_TIER`     | Default tier                     | `standard`   |
| `CHESS_LLM_TRACK_COSTS`      | Enable cost tracking             | `true`       |

See [`chess_llm/config/settings.py`](chess_llm/config/settings.py) for the full list.

## Usage Tracking

```python
from chess_llm import get_tracker

tracker = get_tracker()

# After making requests...
print(f"Total cost: ${tracker.total_cost:.4f}")
print(f"Total tokens: {tracker.total_tokens}")
print(tracker.get_summary())
```

## Security

API keys are loaded from environment variables and **never**:
- Included in `__repr__` output or log messages
- Stored in plain text in configuration files
- Exposed in error messages or tracebacks

See [SECURITY.md](SECURITY.md) for the full security policy.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check .
black --check .
mypy chess_llm/
```

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [KasparChess](https://github.com/vds4321/KasparChess) | Parent-focused chess coaching web application |
| [YourChessDotComCoach](https://github.com/vds4321/YourChessDotComCoach) | Adult chess coaching web application |
| `kaspar_eval` | Prompt evaluation, guardrails testing, and LLM comparison (planned) |

## License

MIT
