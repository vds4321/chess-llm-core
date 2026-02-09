# chess-llm-core

Shared LLM abstraction layer for chess coaching applications.

## Products Using This Library

- [YourChessDotComCoach](https://github.com/vds4321/YourChessDotComCoach) - Adult chess coaching (live at yourchessdotcomcoach.fly.dev)
- [KasparChess](https://github.com/vds4321/KasparChess) - Parent-focused coaching for children (live at [kasparchess.com](https://kasparchess.com))

## Features

- **Protocol-based LLM Provider Abstraction**: Flexible interface supporting multiple LLM providers
- **Model Tier System**: CHEAP/STANDARD/PREMIUM tiers for cost optimization
- **Versioned Prompt Templates**: Reusable prompts for coaching, scouting, and extraction
- **Cost and Usage Tracking**: Monitor LLM usage and costs across requests
- **Provider Registry**: Easy switching between providers (Anthropic, OpenAI, local models)

## Installation

```bash
# Basic installation (no providers)
pip install git+https://github.com/vds4321/chess-llm-core.git

# With Anthropic support
pip install "git+https://github.com/vds4321/chess-llm-core.git#egg=chess-llm-core[anthropic]"

# With all providers
pip install "git+https://github.com/vds4321/chess-llm-core.git#egg=chess-llm-core[all]"

# Development
pip install "git+https://github.com/vds4321/chess-llm-core.git#egg=chess-llm-core[dev]"
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

| Tier | Use Case | Anthropic Model |
|------|----------|-----------------|
| CHEAP | Extraction, classification | Claude 3.5 Haiku |
| STANDARD | Coaching insights, scouting | Claude Sonnet 4 |
| PREMIUM | Comprehensive reports | Claude 3 Opus |

## Prompt Templates

### Coaching
- `MentorInsightsPrompt`: Personalized coaching insights based on game analysis

### Scouting
- `BattlePlanPrompt`: Battle plan for beating a specific opponent

### Extraction
- `KeyAreasExtractionPrompt`: Extract key improvement areas from reports

## Configuration

Environment variables:
- `ANTHROPIC_API_KEY`: Anthropic API key
- `OPENAI_API_KEY`: OpenAI API key (for future use)
- `CHESS_LLM_DEFAULT_PROVIDER`: Default provider (default: "anthropic")
- `CHESS_LLM_DEFAULT_TIER`: Default tier (default: "standard")

## Usage Tracking

```python
from chess_llm import get_tracker

tracker = get_tracker()

# After making requests...
print(f"Total cost: ${tracker.total_cost:.4f}")
print(f"Total tokens: {tracker.total_tokens}")
print(tracker.get_summary())
```

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

## License

MIT
