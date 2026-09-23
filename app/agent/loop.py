"""Interface seam — the agent loop, written out directly.

`run_agent(text, owner_id) -> str` knows nothing about terminals. Telegram
will call it with the Telegram user id as `owner_id`; voice will call it
with a transcript. That is the whole seam — no adapter layer needed.

No framework. The loop below is the entire orchestration: send messages
plus tool schemas, run any tool calls, append the results, repeat until
the model answers with text.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, Optional

from app.agent.prompts import build_system_prompt
from app.agent.tools import TOOL_SCHEMAS, dispatch
from app.config import settings
from app.llm.base import LLMProvider
from app.llm.openai_compat import get_provider

# Conversation context only — so "he" in the next message still resolves.
# Long-term memory lives in the database, not in the context window, which
# is why this can stay tiny and in-process.
#
# TODO: when this becomes a long-lived multi-user service, move to a
# `conversations` table or Redis keyed by owner_id. Nothing else changes.
_HISTORY: dict[str, list[dict[str, Any]]] = {}

OnTool = Callable[[str, dict[str, Any]], None]


def get_history(owner_id: str) -> list[dict[str, Any]]:
    return _HISTORY.setdefault(owner_id, [])


def reset_history(owner_id: str) -> None:
    """Clear the conversation. Memory in the database is untouched."""
    _HISTORY[owner_id] = []


def _trim(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = settings.history_turns * 2  # a turn is one user + one assistant
    return history[-keep:] if len(history) > keep else history


async def run_agent(
    text: str,
    owner_id: str,
    provider: Optional[LLMProvider] = None,
    on_tool: Optional[OnTool] = None,
    source: str = "text",
    audio_path: str | None = None,
) -> str:
    """Handle one user message and return the assistant's reply.

    `on_tool` is display-only: interfaces use it to show activity. It must
    never influence what the loop does.

    `source` and `audio_path` describe how the message arrived. A voice
    note keeps a link to its recording because the transcript becomes an
    IMMUTABLE note: if transcription mangles a sentence — likely when the
    speech mixes English, Malay and Chinese — the audio is the only way
    back to what was actually said.
    """
    context = {"_source": source, "_audio_path": audio_path}
    llm = provider or get_provider()
    history = get_history(owner_id)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        *_trim(history),
        {"role": "user", "content": text},
    ]

    reply = ""
    for _ in range(settings.max_iterations):
        response = await llm.complete(messages, tools=TOOL_SCHEMAS)

        if not response.wants_tools:
            reply = response.content
            break

        messages.append(
            response.raw_message
            or {
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in response.tool_calls
                ],
            }
        )

        for call in response.tool_calls:
            if call.parse_error:
                # Malformed JSON comes back as a tool result, not an
                # exception, so the model gets a chance to fix itself.
                result: dict[str, Any] = {"ok": False, "error": call.parse_error}
            else:
                if on_tool:
                    on_tool(call.name, call.arguments)
                # owner_id is injected here, never read from call.arguments.
                result = await asyncio.to_thread(
                    dispatch, call.name, call.arguments, owner_id, context
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result, default=str),
                }
            )
    else:
        # Ran out of iterations with tools still pending. Say so rather
        # than returning an empty string that reads like a real answer.
        reply = (
            "I got stuck working through that one — it's saved, but ask me again "
            "and I'll try a simpler route."
        )

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    # Only the plain turns are kept. Replaying tool calls would bloat the
    # window and break if a tool result were ever trimmed away from the
    # assistant message that requested it.
    _HISTORY[owner_id] = _trim(history)
    return reply
