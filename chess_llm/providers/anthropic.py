"""
Anthropic (Claude) LLM provider implementation.

This is the primary production provider for all chess coaching applications.
It supports the full Claude model family:

    - Claude 3.5 Haiku  -- fast and cheap (CHEAP tier)
    - Claude Sonnet 4   -- balanced performance (STANDARD tier)
    - Claude 3 Opus     -- highest quality (PREMIUM tier, legacy)

Usage::

    from chess_llm.providers.anthropic import AnthropicProvider
    from chess_llm.config.models import ModelTier

    # Default model from settings
    provider = AnthropicProvider()

    # Specific tier
    provider = AnthropicProvider(tier=ModelTier.CHEAP)

    # Explicit model ID
    provider = AnthropicProvider(model_id="claude-sonnet-4-20250514")

    # Generate a completion
    response = provider.complete("Analyze this chess position...")

Dependencies:
    Requires the ``anthropic`` package::

        pip install chess-llm-core[anthropic]

Security:
    The API key is resolved in this order:
        1. Explicit ``api_key`` parameter
        2. ``Settings.anthropic.api_key`` (from environment)
        3. ``ANTHROPIC_API_KEY`` environment variable directly
    Keys are never logged or included in ``__repr__`` output.

Maintenance:
    When Anthropic releases a new model, add it to
    ``chess_llm.config.models.ANTHROPIC_MODELS`` and update
    ``DEFAULT_MODELS`` if it should be the new default for a tier.
"""

import base64
import os
import time
from typing import Any, Dict, List, Optional

from chess_llm.providers.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    UsageStats,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    TokenLimitError,
)
from chess_llm.config.models import (
    ModelTier,
    ModelConfig,
    get_model_config,
    ANTHROPIC_MODELS,
)
from chess_llm.config.settings import get_settings


# ---------------------------------------------------------------------------
# Image type detection constants
# ---------------------------------------------------------------------------

_IMAGE_SIGNATURES: List[tuple] = [
    (b"\xff\xd8", "image/jpeg"),
    (b"\x89PNG", "image/png"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),
]
_DEFAULT_MEDIA_TYPE = "image/png"


def _detect_media_type(data: bytes) -> str:
    """Detect the MIME type of an image from its magic bytes."""
    for signature, media_type in _IMAGE_SIGNATURES:
        if data[: len(signature)] == signature:
            return media_type
    return _DEFAULT_MEDIA_TYPE


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Wraps the official ``anthropic`` Python SDK and maps responses to the
    library's standardised ``LLMResponse`` format.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialise the provider and validate credentials.

        Args:
            model_id: Specific model to use (overrides *tier*).
            tier:     Model tier -- ignored when *model_id* is set.
            api_key:  API key.  Falls back to settings / environment.
            **kwargs: Forwarded to the ``anthropic.Anthropic`` client.

        Raises:
            ValueError:           Unknown model ID.
            AuthenticationError:  No API key found anywhere.
            ImportError:          ``anthropic`` package not installed.
        """
        # -- Resolve model config -------------------------------------------
        if model_id:
            if model_id not in ANTHROPIC_MODELS:
                raise ValueError(f"Unknown Anthropic model: {model_id}")
            self._model_config = ANTHROPIC_MODELS[model_id]
        elif tier:
            self._model_config = get_model_config(tier, provider="anthropic")
        else:
            settings = get_settings()
            self._model_config = get_model_config(
                settings.default_tier, provider="anthropic"
            )

        # -- Resolve API key ------------------------------------------------
        if api_key is None:
            settings = get_settings()
            api_key = settings.anthropic.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise AuthenticationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter.",
                provider="anthropic",
            )

        super().__init__(
            model_id=self._model_config.model_id,
            api_key=api_key,
            **kwargs,
        )

        # -- Initialise SDK client ------------------------------------------
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key, **kwargs)
            self._async_client = None  # Lazy-init on first async call
            self._client_kwargs = kwargs  # Save for async client creation
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install chess-llm-core[anthropic]"
            )

    # -----------------------------------------------------------------------
    # Protocol properties
    # -----------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return self._model_config.display_name

    @property
    def supports_vision(self) -> bool:
        return self._model_config.supports_vision

    @property
    def max_context_tokens(self) -> int:
        return self._model_config.max_context_tokens

    @property
    def max_output_tokens(self) -> int:
        return self._model_config.max_output_tokens

    @property
    def model_config(self) -> ModelConfig:
        """Full ``ModelConfig`` for the active model."""
        return self._model_config

    def get_cost_per_token(self) -> tuple[float, float]:
        """Return ``(input_cost/M, output_cost/M)`` in USD."""
        return (
            self._model_config.input_cost_per_million,
            self._model_config.output_cost_per_million,
        )

    # -----------------------------------------------------------------------
    # Text completion
    # -----------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a text completion via the Anthropic Messages API.

        Args:
            prompt: User message text.
            config: Per-request configuration (uses defaults if ``None``).

        Returns:
            Standardised ``LLMResponse``.
        """
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=[{"role": "user", "content": prompt}],
                config=config,
            )

            response = self._client.messages.create(**request_kwargs)
            latency_ms = (time.time() - start_time) * 1000

            return self._build_response(response, latency_ms)

        except Exception as e:
            self._handle_error(e)

    # -----------------------------------------------------------------------
    # Vision completion
    # -----------------------------------------------------------------------

    def complete_with_images(
        self,
        prompt: str,
        images: List[bytes],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion that includes image context.

        Images are base64-encoded and sent as inline content blocks.

        Args:
            prompt: User message text.
            images: List of images as raw bytes (PNG / JPEG / GIF / WebP).
            config: Per-request configuration.

        Raises:
            NotImplementedError: If the active model lacks vision support.
        """
        if not self.supports_vision:
            raise NotImplementedError(
                f"Model {self._model_id} does not support vision"
            )

        config = config or self._get_default_config()
        start_time = time.time()

        try:
            # Build multi-part content: images first, then text
            content: List[Dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _detect_media_type(image_data),
                        "data": base64.b64encode(image_data).decode("utf-8"),
                    },
                }
                for image_data in images
            ]
            content.append({"type": "text", "text": prompt})

            request_kwargs = self._build_request_kwargs(
                messages=[{"role": "user", "content": content}],
                config=config,
            )

            response = self._client.messages.create(**request_kwargs)
            latency_ms = (time.time() - start_time) * 1000

            return self._build_response(response, latency_ms)

        except Exception as e:
            self._handle_error(e)

    # -----------------------------------------------------------------------
    # Messages-list completion (sync)
    # -----------------------------------------------------------------------

    def complete_messages(
        self,
        messages: List[Dict[str, Any]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion from a messages list (multi-turn).

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            config:   Per-request configuration.
        """
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=messages,
                config=config,
            )

            response = self._client.messages.create(**request_kwargs)
            latency_ms = (time.time() - start_time) * 1000

            return self._build_response(response, latency_ms)

        except Exception as e:
            self._handle_error(e)

    # -----------------------------------------------------------------------
    # Async completions (native AsyncAnthropic)
    # -----------------------------------------------------------------------

    def _get_async_client(self):
        """Lazily initialise the async Anthropic client."""
        if self._async_client is None:
            import anthropic

            self._async_client = anthropic.AsyncAnthropic(
                api_key=self._api_key, **self._client_kwargs
            )
        return self._async_client

    async def acomplete(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Async text completion using ``AsyncAnthropic``.

        Native async -- no thread pool needed.
        """
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=[{"role": "user", "content": prompt}],
                config=config,
            )

            client = self._get_async_client()
            response = await client.messages.create(**request_kwargs)
            latency_ms = (time.time() - start_time) * 1000

            return self._build_response(response, latency_ms)

        except Exception as e:
            self._handle_error(e)

    async def acomplete_messages(
        self,
        messages: List[Dict[str, Any]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Async messages-list completion using ``AsyncAnthropic``.

        Native async -- no thread pool needed.
        """
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=messages,
                config=config,
            )

            client = self._get_async_client()
            response = await client.messages.create(**request_kwargs)
            latency_ms = (time.time() - start_time) * 1000

            return self._build_response(response, latency_ms)

        except Exception as e:
            self._handle_error(e)

    # -----------------------------------------------------------------------
    # Token estimation
    # -----------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 characters per token for English)."""
        return len(text) // 4

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_request_kwargs(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
    ) -> Dict[str, Any]:
        """
        Assemble the keyword arguments for ``client.messages.create()``.

        Only includes optional parameters when they have meaningful values,
        avoiding API rejections from ``None`` or empty fields.
        """
        kwargs: Dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": min(config.max_tokens, self.max_output_tokens),
            "messages": messages,
        }

        # Anthropic API does not allow both temperature and top_p
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        else:
            kwargs["temperature"] = config.temperature

        if config.system_prompt:
            kwargs["system"] = config.system_prompt
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        kwargs.update(config.extra)
        return kwargs

    def _build_response(self, response: Any, latency_ms: float) -> LLMResponse:
        """Convert a raw Anthropic API response to an ``LLMResponse``."""
        usage = self._calculate_cost(
            UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=(
                    response.usage.input_tokens + response.usage.output_tokens
                ),
            )
        )

        content = ""
        if response.content:
            content = response.content[0].text

        return LLMResponse(
            content=content,
            model=self._model_id,
            provider=self.provider_id,
            usage=usage,
            finish_reason=response.stop_reason or "stop",
            latency_ms=latency_ms,
            raw_response={
                "id": response.id,
                "type": response.type,
                "model": response.model,
                "stop_reason": response.stop_reason,
            },
        )

    def _handle_error(self, error: Exception) -> None:
        """
        Translate Anthropic SDK exceptions into chess-llm-core error types.

        This ensures callers only need to handle our exception hierarchy
        regardless of which provider is active.
        """
        import anthropic

        error_msg = str(error)

        if isinstance(error, anthropic.RateLimitError):
            retry_after = None
            if hasattr(error, "response") and error.response:
                raw = error.response.headers.get("retry-after")
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        retry_after = None
            raise RateLimitError(
                f"Anthropic rate limit exceeded: {error_msg}",
                provider="anthropic",
                retry_after=retry_after,
            )

        if isinstance(error, anthropic.AuthenticationError):
            raise AuthenticationError(
                f"Anthropic authentication failed: {error_msg}",
                provider="anthropic",
            )

        if isinstance(error, anthropic.BadRequestError):
            if "token" in error_msg.lower():
                raise TokenLimitError(
                    f"Token limit exceeded: {error_msg}",
                    provider="anthropic",
                )
            raise ProviderError(
                f"Anthropic request error: {error_msg}",
                provider="anthropic",
            )

        if isinstance(error, anthropic.APIError):
            raise ProviderError(
                f"Anthropic API error: {error_msg}",
                provider="anthropic",
            )

        raise ProviderError(
            f"Unexpected error: {error_msg}",
            provider="anthropic",
        )
