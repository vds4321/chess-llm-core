# Security Policy

## Overview

`chess-llm-core` is a shared library that communicates with third-party LLM APIs on behalf of consuming applications.  This document describes the security measures in place to protect API keys, user data, and consuming applications.

## API Key Protection

### How keys are handled

- API keys are loaded **exclusively from environment variables** (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
- Keys are **never hard-coded** in source files, configuration files, or test fixtures.
- `ProviderSettings.__repr__()` and `BaseLLMProvider.__repr__()` mask keys so that `print()`, logging, and tracebacks never expose them.
- The `.gitignore` excludes `.env` files to prevent accidental commits.

### Recommendations for consuming applications

- Store API keys in your deployment platform's secrets manager (e.g. Fly.io secrets, Heroku config vars, AWS Secrets Manager).
- Never pass API keys as command-line arguments (they appear in process listings).
- Rotate keys periodically and immediately if a compromise is suspected.

## User Data Protection

### Prompt content

Prompt templates may contain usernames and game statistics.  This data is:

- Sent to the LLM provider's API over HTTPS.
- **Not logged by default**.  The `log_requests` and `log_responses` settings are `False` by default.  Enable them only in development environments with appropriate access controls.
- **Not persisted** by this library.  The `UsageTracker` stores token counts and costs but **not** prompt or response content.

### PII considerations

- Player usernames are public chess.com / Lichess handles, not private identifiers.
- No email addresses, real names, passwords, or financial data are processed by this library.
- Game statistics are derived from publicly available chess platform data.

## Dependency Security

- The library has a single required dependency (`pydantic`).
- Provider SDKs (`anthropic`, `openai`) are optional and only imported when explicitly installed.
- All dependencies are pinned to minimum versions; consuming applications should use a lock file (e.g. `requirements.txt` with hashes, or `poetry.lock`) for reproducible builds.

## Error Handling

- Error messages from provider APIs are wrapped in library-specific exception types (`ProviderError`, `RateLimitError`, `AuthenticationError`, `TokenLimitError`).
- Error messages **never include the API key**.  Authentication errors report that the key is invalid without echoing it.
- The `raw_response` field in `LLMResponse` stores only non-sensitive metadata (request ID, model name, stop reason).

## Rate Limiting

- The `Settings` class provides `requests_per_minute` and `tokens_per_minute` fields for client-side rate limiting.
- The `UsageTracker` supports budget limits (`budget_limit_usd`) to prevent runaway costs.
- The Anthropic provider automatically translates API rate-limit errors into `RateLimitError` exceptions with `retry_after` hints.

## Reporting a Vulnerability

If you discover a security vulnerability in this library, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Contact the maintainer directly at the email address listed in `pyproject.toml`.
3. Include a description of the vulnerability and steps to reproduce.
4. Allow reasonable time for a fix before public disclosure.
