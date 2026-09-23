"""Model seam — one internal format that every provider is flattened into.

No provider-specific field may cross this line. The agent loop and the
tools see only `LLMResponse` and `ToolCall`; swapping DeepSeek for Groq,
OpenRouter, Together or a local Ollama must not touch a line of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """One requested tool call, already parsed into plain Python."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model emitted JSON we couldn't parse. The loop returns
    # this text to the model as a tool result instead of crashing —
    # smaller models produce bad JSON often enough to matter.
    parse_error: str | None = None


@dataclass
class LLMResponse:
    """A single assistant turn: text, tool calls, or both."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Opaque provider payload, carried so the loop can echo the exact
    # assistant message back in history. Never inspected above this layer.
    raw_message: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMProvider(Protocol):
    """The entire contract. One method is enough; resist adding more."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse: ...
