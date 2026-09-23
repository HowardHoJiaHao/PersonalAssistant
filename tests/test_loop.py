"""Agent loop against a scripted fake provider. No API key, no network.

The fake returns a fixed sequence of turns, which is exactly what the
model seam buys us: the loop can be tested without a model at all.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.agent.loop import get_history, reset_history, run_agent
from app.llm.base import LLMProvider, LLMResponse, ToolCall

OWNER = "test-owner"


class ScriptedProvider:
    """Replays a list of LLMResponses and records what it was asked."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return self.script.pop(0)


def _tool_turn(call_id: str, name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        raw_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        },
    )


def test_scripted_provider_satisfies_the_protocol():
    assert isinstance(ScriptedProvider([]), LLMProvider)


def test_loop_runs_tools_in_order_and_returns_text():
    reset_history(OWNER)
    seen: list[tuple[str, dict[str, Any]]] = []

    provider = ScriptedProvider(
        [
            _tool_turn(
                "c1",
                "save_note",
                {
                    "raw_text": "had lunch with Peter at Village Park today, he's really into matcha now",
                    "event_date": "2026-09-23",
                    "location": "Village Park",
                },
            ),
            _tool_turn("c2", "find_person", {"name": "Peter"}),
            _tool_turn("c3", "upsert_person", {"display_name": "Peter", "aliases": ["peter"]}),
            _tool_turn(
                "c4",
                "save_facts",
                {
                    "facts": [
                        {
                            "person_id": 1,
                            "category": "food",
                            "key": "matcha",
                            "value": "likes matcha a lot",
                            "reason": "got into it in Kyoto",
                            "source_note_id": 1,
                        }
                    ]
                },
            ),
            LLMResponse(content="Noted — Peter's into matcha now."),
        ]
    )

    reply = asyncio.run(
        run_agent(
            "had lunch with Peter at Village Park today, he's really into matcha now",
            OWNER,
            provider=provider,
            on_tool=lambda name, args: seen.append((name, args)),
        )
    )

    assert [name for name, _ in seen] == [
        "save_note",
        "find_person",
        "upsert_person",
        "save_facts",
    ]
    assert reply == "Noted — Peter's into matcha now."
    print("\ntool order:", [n for n, _ in seen])
    print("reply:", reply)

    # The turn is recorded in history; tool churn is not.
    history = get_history(OWNER)
    assert history[-2]["role"] == "user"
    assert history[-1] == {"role": "assistant", "content": reply}
    assert all(m["role"] in {"user", "assistant"} for m in history)

    # The system prompt is sent every call, and tool results were fed back.
    first_call, last_call = provider.calls[0], provider.calls[-1]
    assert first_call[0]["role"] == "system"
    assert "VERBATIM" in first_call[0]["content"]
    assert sum(1 for m in last_call if m["role"] == "tool") == 4

    # The note really landed, verbatim, via the dispatcher.
    from app.agent.tools import dispatch

    notes = dispatch("search_notes", {"query": "Village Park"}, OWNER)["notes"]
    assert notes[0]["raw_text"].startswith("had lunch with Peter")


def test_history_carries_between_turns():
    reset_history(OWNER)
    p1 = ScriptedProvider([LLMResponse(content="Got it.")])
    asyncio.run(run_agent("Peter loves matcha", OWNER, provider=p1))

    p2 = ScriptedProvider([LLMResponse(content="At Grab.")])
    asyncio.run(run_agent("where does he work?", OWNER, provider=p2))

    sent = p2.calls[0]
    assert sent[1] == {"role": "user", "content": "Peter loves matcha"}
    assert sent[2] == {"role": "assistant", "content": "Got it."}
    assert sent[-1] == {"role": "user", "content": "where does he work?"}


def test_history_is_trimmed_and_reset_keeps_memory():
    from app.agent.tools import dispatch
    from app.config import settings

    reset_history(OWNER)
    for i in range(settings.history_turns + 5):
        provider = ScriptedProvider([LLMResponse(content=f"ok {i}")])
        asyncio.run(run_agent(f"message {i}", OWNER, provider=provider))

    assert len(get_history(OWNER)) == settings.history_turns * 2

    dispatch("upsert_person", {"display_name": "Zoe"}, OWNER)
    reset_history(OWNER)
    assert get_history(OWNER) == []
    # /new clears the conversation, never the database.
    assert dispatch("find_person", {"name": "Zoe"}, OWNER)["match_count"] == 1


def test_malformed_tool_json_is_returned_to_the_model_not_raised():
    reset_history(OWNER)
    bad = LLMResponse(
        tool_calls=[
            ToolCall(
                id="c1",
                name="save_note",
                arguments={},
                parse_error="Could not parse your tool arguments as JSON.",
            )
        ],
        raw_message={"role": "assistant", "content": None, "tool_calls": []},
    )
    provider = ScriptedProvider([bad, LLMResponse(content="Saved.")])

    seen: list[str] = []
    reply = asyncio.run(
        run_agent("Peter likes matcha", OWNER, provider=provider, on_tool=lambda n, a: seen.append(n))
    )

    assert reply == "Saved."
    assert seen == [], "a call that failed to parse must not reach the dispatcher"
    tool_msgs = [m for m in provider.calls[-1] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["ok"] is False
    print("\nmalformed JSON fed back to model:", tool_msgs[0]["content"][:80])


def test_iteration_cap_stops_a_tool_loop():
    from app.config import settings

    reset_history(OWNER)
    provider = ScriptedProvider(
        [_tool_turn(f"c{i}", "list_people", {}) for i in range(settings.max_iterations)]
    )
    reply = asyncio.run(run_agent("hello", OWNER, provider=provider))

    assert len(provider.calls) == settings.max_iterations
    assert "stuck" in reply.lower()
    assert get_history(OWNER)[-1]["content"] == reply


def test_owner_id_is_not_reachable_from_tool_arguments():
    """Even if the model names another owner, the loop injects the real one."""
    from app.agent.tools import dispatch

    reset_history("owner-x")
    dispatch("upsert_person", {"display_name": "Private Person"}, "owner-x")

    provider = ScriptedProvider(
        [_tool_turn("c1", "list_people", {"owner_id": "owner-x"}), LLMResponse(content="none")]
    )
    asyncio.run(run_agent("who do I know?", "owner-y", provider=provider))

    tool_msgs = [m for m in provider.calls[-1] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["people"] == []
