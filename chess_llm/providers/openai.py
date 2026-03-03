"""
OpenAI (GPT) LLM provider implementation.

Supports the GPT model family via the official ``openai`` Python SDK:

    - GPT-4o Mini  -- fast and cheap (CHEAP tier)
    - GPT-4o       -- balanced performance (STANDARD tier)

Usage::

    from chess_llm.providers.openai import OpenAIProvider
    from chess_llm.config.models import ModelTier

    provider = OpenAIProvider(tier=ModelTier.CHEAP)
    response = provider.complete("Analyze this chess position...")

    # Async
    response = await provider.acomplete("Analyze this chess position...")

Dependencies:
    Requires the ``openai`` package::

        pip install openai

Security:
    The API key is resolved in this order:
        1. Explicit ``api_key`` parameter
        2. ``OPENAI_API_KEY`` environment variable
"""

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
    OPENAI_MODELS,
)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT provider.

    Wraps the official ``openai`` Python SDK and maps responses to the
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
        Initialise the provider.

        Args:
            model_id: Specific model to use (overrides *tier*).
            tier:     Model tier -- ignored when *model_id* is set.
            api_key:  API key.  Falls back to environment.
            **kwargs: Forwarded to the ``openai.OpenAI`` client.
        """
        # -- Resolve model config -------------------------------------------
        if model_id:
            if model_id not in OPENAI_MODELS:
                raise ValueError(f"Unknown OpenAI model: {model_id}")
            self._model_config = OPENAI_MODELS[model_id]
        elif tier:
            self._model_config = get_model_config(tier, provider="openai")
        else:
            self._model_config = get_model_config(ModelTier.STANDARD, provider="openai")

        if self._model_config is None:
            raise ValueError("Could not resolve OpenAI model configuration")

        # -- Resolve API key ------------------------------------------------
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise AuthenticationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter.",
                provider="openai",
            )

        super().__init__(
            model_id=self._model_config.model_id,
            api_key=api_key,
            **kwargs,
        )

        # -- Initialise SDK client ------------------------------------------
        try:
            import openai

            self._client = openai.OpenAI(api_key=api_key, **kwargs)
            self._async_client = None  # Lazy-init on first async call
            self._client_kwargs = kwargs
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

    # -----------------------------------------------------------------------
    # Protocol properties
    # -----------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return "openai"

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
        return (
            self._model_config.input_cost_per_million,
            self._model_config.output_cost_per_million,
        )

    # -----------------------------------------------------------------------
    # Text completion (sync)
    # -----------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a text completion via the OpenAI Chat Completions API."""
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=[{"role": "user", "content": prompt}],
                config=config,
            )
            response = self._client.chat.completions.create(**request_kwargs)
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
        """Generate a completion from a messages list."""
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=messages,
                config=config,
            )
            response = self._client.chat.completions.create(**request_kwargs)
            latency_ms = (time.time() - start_time) * 1000
            return self._build_response(response, latency_ms)

        except Exception as e:
            self._handle_error(e)

    # -----------------------------------------------------------------------
    # Async completions
    # -----------------------------------------------------------------------

    def _get_async_client(self):
        """Lazily initialise the async OpenAI client."""
        if self._async_client is None:
            import openai

            self._async_client = openai.AsyncOpenAI(
                api_key=self._api_key, **self._client_kwargs
            )
        return self._async_client

    async def acomplete(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Async text completion using ``AsyncOpenAI``."""
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=[{"role": "user", "content": prompt}],
                config=config,
            )
            client = self._get_async_client()
            response = await client.chat.completions.create(**request_kwargs)
            latency_ms = (time.time() - start_time) * 1000
            return self._build_response(response, latency_ms)

        except Exception as e:
            self._handle_error(e)

    async def acomplete_messages(
        self,
        messages: List[Dict[str, Any]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Async messages-list completion using ``AsyncOpenAI``."""
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            request_kwargs = self._build_request_kwargs(
                messages=messages,
                config=config,
            )
            client = self._get_async_client()
            response = await client.chat.completions.create(**request_kwargs)
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
        """Assemble kwargs for ``chat.completions.create()``."""
        all_messages = list(messages)

        # Prepend system message if provided
        if config.system_prompt:
            all_messages = [
                {"role": "system", "content": config.system_prompt}
            ] + all_messages

        kwargs: Dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": min(config.max_tokens, self.max_output_tokens),
            "messages": all_messages,
            "temperature": config.temperature,
        }

        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        kwargs.update(config.extra)
        return kwargs

    def _build_response(self, response: Any, latency_ms: float) -> LLMResponse:
        """Convert a raw OpenAI API response to an ``LLMResponse``."""
        usage_data = response.usage
        usage = self._calculate_cost(
            UsageStats(
                input_tokens=usage_data.prompt_tokens if usage_data else 0,
                output_tokens=usage_data.completion_tokens if usage_data else 0,
                total_tokens=usage_data.total_tokens if usage_data else 0,
            )
        )

        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        finish_reason = "stop"
        if response.choices and response.choices[0].finish_reason:
            finish_reason = response.choices[0].finish_reason

        return LLMResponse(
            content=content,
            model=self._model_id,
            provider=self.provider_id,
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response={
                "id": response.id,
                "model": response.model,
                "finish_reason": finish_reason,
            },
        )

    def _handle_error(self, error: Exception) -> None:
        """Translate OpenAI SDK exceptions into chess-llm-core error types."""
        try:
            import openai as openai_module
        except ImportError:
            raise ProviderError(str(error), provider="openai")

        error_msg = str(error)

        if isinstance(error, openai_module.RateLimitError):
            retry_after = None
            if hasattr(error, "response") and error.response:
                raw = error.response.headers.get("retry-after")
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        retry_after = None
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {error_msg}",
                provider="openai",
                retry_after=retry_after,
            )

        if isinstance(error, openai_module.AuthenticationError):
            raise AuthenticationError(
                f"OpenAI authentication failed: {error_msg}",
                provider="openai",
            )

        if isinstance(error, openai_module.BadRequestError):
            if "token" in error_msg.lower() or "context" in error_msg.lower():
                raise TokenLimitError(
                    f"Token limit exceeded: {error_msg}",
                    provider="openai",
                )
            raise ProviderError(
                f"OpenAI request error: {error_msg}",
                provider="openai",
            )

        if isinstance(error, openai_module.APIError):
            raise ProviderError(
                f"OpenAI API error: {error_msg}",
                provider="openai",
            )

        raise ProviderError(
            f"Unexpected error: {error_msg}",
            provider="openai",
        )
