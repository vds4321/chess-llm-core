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
│   ├── openai.py   # OpenAI GPT implementation (production-ready)
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

# Generate response (sync)
response = provider.complete(prompt.render())
print(response.content)
```

### Multi-Turn Messages

```python
# Use a messages list for multi-turn conversations
response = provider.complete_messages([
    {"role": "user", "content": "Analyze this opening: 1. e4 e5 2. Nf3 Nc6"},
])
```

### Async API

Both Anthropic and OpenAI providers have native async implementations (not thread-pool wrappers):

```python
import asyncio
from chess_llm import get_provider, ModelTier

async def main():
    provider = get_provider("anthropic", tier=ModelTier.STANDARD)

    # Async text completion
    response = await provider.acomplete("Analyze this chess position...")

    # Async multi-turn
    response = await provider.acomplete_messages([
        {"role": "user", "content": "What should I play after 1. d4?"},
    ])

asyncio.run(main())
```

### Vision (Image Analysis)

Providers that support vision (Claude and GPT-4o) can analyze board positions:

```python
with open("board.png", "rb") as f:
    image_data = f.read()

response = provider.complete_with_images(
    prompt="What is the best move in this position?",
    images=[image_data],
)
```

## Model Tiers

### Anthropic

| Tier       | Use Case                    | Model              | Cost (input/output per 1M tokens) |
|------------|-----------------------------|--------------------|-----------------------------------|
| `CHEAP`    | Extraction, classification  | Claude Haiku 4.5   | $1.00 / $5.00                     |
| `STANDARD` | Coaching, scouting, analysis| Claude Sonnet 4.6  | $3.00 / $15.00                    |
| `PREMIUM`  | Comprehensive reports       | Claude Opus 4.6    | $5.00 / $25.00                    |

### OpenAI

| Tier       | Use Case                    | Model              | Cost (input/output per 1M tokens) |
|------------|-----------------------------|--------------------|-----------------------------------|
| `CHEAP`    | Extraction, classification  | GPT-4o Mini        | $0.15 / $0.60                     |
| `STANDARD` | Coaching, scouting, analysis| GPT-4o             | $2.50 / $10.00                    |

All models support vision (image inputs). Context windows: Anthropic 200K tokens, OpenAI 128K tokens.

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
| `ANTHROPIC_API_KEY`          | Anthropic API key                | *required for Anthropic* |
| `OPENAI_API_KEY`             | OpenAI API key                   | *required for OpenAI* |
| `CHESS_LLM_DEFAULT_PROVIDER` | Default provider                 | `anthropic`  |
| `CHESS_LLM_DEFAULT_TIER`     | Default tier                     | `standard`   |
| `CHESS_LLM_TRACK_COSTS`      | Enable cost tracking             | `true`       |
| `CHESS_LLM_LOG_REQUESTS`     | Log outgoing LLM requests        | `false`      |
| `CHESS_LLM_LOG_RESPONSES`    | Log LLM responses                | `false`      |
| `ANTHROPIC_API_BASE`         | Custom Anthropic API base URL    | —            |
| `OPENAI_API_BASE`            | Custom OpenAI API base URL       | —            |

### Per-Provider Settings

Each provider also has configurable `timeout_seconds` (default 60), `max_retries` (default 3), and rate limits (`requests_per_minute` default 60, `tokens_per_minute` default 100K). A `cost_warning_threshold_usd` (default $1.00) logs warnings for expensive requests.

Settings can also be configured programmatically:

```python
from chess_llm import get_settings, Settings

# Read current settings
settings = get_settings()

# Or configure programmatically
from chess_llm.config.settings import configure_settings, reset_settings
configure_settings(custom_settings)
reset_settings()  # clear cached singleton
```

See [`chess_llm/config/settings.py`](chess_llm/config/settings.py) for the full list.

## Provider Registry

```python
from chess_llm import get_provider, list_providers, ModelTier
from chess_llm.providers.registry import get_provider_for_tier, is_provider_available

# List registered providers
print(list_providers())  # ['anthropic', 'openai', 'local']

# Check if a provider's SDK is installed
if is_provider_available("openai"):
    provider = get_provider("openai", tier=ModelTier.STANDARD)

# Convenience: get provider by tier (uses default provider)
provider = get_provider_for_tier(ModelTier.CHEAP)
```

### Model Lookup Helpers

```python
from chess_llm.config.models import (
    get_model_config, get_default_model,
    get_models_by_tier, get_cheapest_model,
)

# Get config for a tier+provider
config = get_model_config(ModelTier.STANDARD, provider="anthropic")

# Find cheapest model meeting requirements
config = get_cheapest_model(min_quality=0.6, requires_vision=True)
```

## Error Handling

All provider errors inherit from `ProviderError`:

```python
from chess_llm import ProviderError, RateLimitError, AuthenticationError, TokenLimitError

try:
    response = provider.complete("...")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except AuthenticationError:
    print("Invalid API key")
except TokenLimitError as e:
    print(f"Requested {e.requested_tokens} tokens, max is {e.max_tokens}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

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
