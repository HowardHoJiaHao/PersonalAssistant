"""One adapter for every OpenAI-compatible host.

DeepSeek, Groq, OpenRouter, Together, Fireworks and local vLLM/Ollama all
speak this wire format including function calling, so they differ by
base_url and model name alone — which is why there is one class here and
not one per vendor.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.llm.base import LLMProvider, LLMResponse, ToolCall


class OpenAICompatibleProvider:
    """Implements LLMProvider against any OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.temperature = (
            settings.llm_temperature if temperature is None else temperature
        )
        self._client = AsyncOpenAI(
            api_key=api_key or settings.llm_api_key or "missing-key",
            base_url=base_url or settings.llm_base_url,
            timeout=timeout or settings.llm_timeout_seconds,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            raw_args = call.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
                if not isinstance(parsed, dict):
                    raise ValueError("arguments must be a JSON object")
                calls.append(
                    ToolCall(id=call.id, name=call.function.name, arguments=parsed)
                )
            except (json.JSONDecodeError, ValueError) as exc:
                # Don't crash the loop on malformed JSON. Hand the error
                # back as a tool result so the model can retry — smaller
                # models emit bad arguments occasionally.
                calls.append(
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments={},
                        parse_error=(
                            f"Could not parse your tool arguments as JSON ({exc}). "
                            f"You sent: {raw_args[:400]}. "
                            "Re-send this tool call with valid JSON."
                        ),
                    )
                )

        return LLMResponse(
            content=message.content or "",
            tool_calls=calls,
            # model_dump keeps the assistant turn replayable in history
            # without the loop ever reading a provider-shaped field.
            raw_message=message.model_dump(exclude_none=True),
        )


def get_provider() -> LLMProvider:
    """Factory reading config. The one place a provider is chosen.

    A plugin registry would be premature here — adding a provider that
    is not OpenAI-compatible means writing a class and one elif.
    """
    return OpenAICompatibleProvider()
