"""
Anthropic (Claude) LLM provider implementation.

Supports all Claude models including:
- Claude 3.5 Haiku (fast, cheap)
- Claude 3.5 Sonnet (balanced)
- Claude 3 Opus (highest quality)

Usage:
    from chess_llm.providers.anthropic import AnthropicProvider
    from chess_llm.config.models import ModelTier

    # Create provider with default model
    provider = AnthropicProvider()

    # Create provider for specific tier
    provider = AnthropicProvider(tier=ModelTier.CHEAP)  # Uses Haiku

    # Create provider with specific model
    provider = AnthropicProvider(model_id="claude-sonnet-4-20250514")

    # Generate completion
    response = provider.complete("Analyze this chess position...")
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


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider implementation.

    Requires the 'anthropic' package: pip install chess-llm-core[anthropic]
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        tier: Optional[ModelTier] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            model_id: Specific model ID to use (e.g., "claude-sonnet-4-20250514")
            tier: Model tier to use (CHEAP, STANDARD, PREMIUM). Ignored if model_id is set.
            api_key: API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            **kwargs: Additional options passed to the Anthropic client
        """
        # Determine model
        if model_id:
            if model_id not in ANTHROPIC_MODELS:
                raise ValueError(f"Unknown Anthropic model: {model_id}")
            self._model_config = ANTHROPIC_MODELS[model_id]
        elif tier:
            self._model_config = get_model_config(tier, provider="anthropic")
        else:
            # Use settings default
            settings = get_settings()
            self._model_config = get_model_config(settings.default_tier, provider="anthropic")

        # Get API key
        if api_key is None:
            settings = get_settings()
            api_key = settings.anthropic.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise AuthenticationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="anthropic",
            )

        super().__init__(model_id=self._model_config.model_id, api_key=api_key, **kwargs)

        # Initialize Anthropic client
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key, **kwargs)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install chess-llm-core[anthropic]"
            )

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
        """Get the full model configuration."""
        return self._model_config

    def get_cost_per_token(self) -> tuple[float, float]:
        """Get cost per million tokens (input, output)."""
        return (
            self._model_config.input_cost_per_million,
            self._model_config.output_cost_per_million,
        )

    def complete(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion using Claude.

        Args:
            prompt: The user prompt/message
            config: Optional configuration

        Returns:
            LLMResponse with the generated content
        """
        config = config or self._get_default_config()
        start_time = time.time()

        try:
            # Build messages
            messages = [{"role": "user", "content": prompt}]

            # Build request kwargs - only include system if present
            request_kwargs: Dict[str, Any] = {
                "model": self._model_id,
                "max_tokens": min(config.max_tokens, self.max_output_tokens),
                "messages": messages,
                "temperature": config.temperature,
            }

            # Only add system prompt if provided (API rejects None or empty)
            if config.system_prompt:
                request_kwargs["system"] = config.system_prompt

            # Only add optional params if they have values
            if config.top_p is not None:
                request_kwargs["top_p"] = config.top_p
            if config.stop_sequences:
                request_kwargs["stop_sequences"] = config.stop_sequences

            # Add any extra kwargs
            request_kwargs.update(config.extra)

            # Make request
            response = self._client.messages.create(**request_kwargs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Build usage stats
            usage = UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            usage = self._calculate_cost(usage)

            # Extract content
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

        except Exception as e:
            self._handle_error(e)

    def complete_with_images(
        self,
        prompt: str,
        images: List[bytes],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion with image context.

        Args:
            prompt: The user prompt/message
            images: List of images as bytes (PNG/JPEG)
            config: Optional configuration

        Returns:
            LLMResponse with the generated content
        """
        if not self.supports_vision:
            raise NotImplementedError(
                f"Model {self._model_id} does not support vision"
            )

        config = config or self._get_default_config()
        start_time = time.time()

        try:
            # Build content with images
            content = []

            # Add images first
            for image_data in images:
                # Detect image type
                media_type = "image/png"
                if image_data[:2] == b"\xff\xd8":
                    media_type = "image/jpeg"
                elif image_data[:4] == b"\x89PNG":
                    media_type = "image/png"
                elif image_data[:4] == b"GIF8":
                    media_type = "image/gif"
                elif image_data[:4] == b"RIFF":
                    media_type = "image/webp"

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_data).decode("utf-8"),
                    },
                })

            # Add text prompt
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            # Build request kwargs - only include system if present
            request_kwargs: Dict[str, Any] = {
                "model": self._model_id,
                "max_tokens": min(config.max_tokens, self.max_output_tokens),
                "messages": messages,
                "temperature": config.temperature,
            }

            # Only add system prompt if provided
            if config.system_prompt:
                request_kwargs["system"] = config.system_prompt

            # Add any extra kwargs
            request_kwargs.update(config.extra)

            # Make request
            response = self._client.messages.create(**request_kwargs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Build usage stats
            usage = UsageStats(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            usage = self._calculate_cost(usage)

            # Extract content
            text_content = ""
            if response.content:
                text_content = response.content[0].text

            return LLMResponse(
                content=text_content,
                model=self._model_id,
                provider=self.provider_id,
                usage=usage,
                finish_reason=response.stop_reason or "stop",
                latency_ms=latency_ms,
            )

        except Exception as e:
            self._handle_error(e)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Claude models.

        Uses Anthropic's approximate formula: ~4 characters per token.
        """
        # Claude uses a similar tokenization to GPT models
        # Rough estimate: 4 characters per token for English
        return len(text) // 4

    def _handle_error(self, error: Exception) -> None:
        """Convert Anthropic errors to our error types."""
        import anthropic

        error_msg = str(error)

        if isinstance(error, anthropic.RateLimitError):
            # Extract retry-after if available
            retry_after = None
            if hasattr(error, "response") and error.response:
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    try:
                        retry_after = float(retry_after)
                    except ValueError:
                        retry_after = None

            raise RateLimitError(
                f"Anthropic rate limit exceeded: {error_msg}",
                provider="anthropic",
                retry_after=retry_after,
            )

        elif isinstance(error, anthropic.AuthenticationError):
            raise AuthenticationError(
                f"Anthropic authentication failed: {error_msg}",
                provider="anthropic",
            )

        elif isinstance(error, anthropic.BadRequestError):
            if "token" in error_msg.lower():
                raise TokenLimitError(
                    f"Token limit exceeded: {error_msg}",
                    provider="anthropic",
                )
            raise ProviderError(
                f"Anthropic request error: {error_msg}",
                provider="anthropic",
            )

        elif isinstance(error, anthropic.APIError):
            raise ProviderError(
                f"Anthropic API error: {error_msg}",
                provider="anthropic",
            )

        else:
            raise ProviderError(
                f"Unexpected error: {error_msg}",
                provider="anthropic",
            )
