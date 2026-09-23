"""Telegram interface — the parts testable without a bot token.

The allow-list is a security boundary: this bot writes to a private
database about real people, so it gets tests even though the transport
does not.
"""

from __future__ import annotations

import pytest

import app.interfaces.telegram as tg
from app.config import Settings


def _with(monkeypatch, **overrides):
    monkeypatch.setattr(tg, "settings", Settings(**overrides))


def test_unlisted_users_are_refused(monkeypatch):
    _with(monkeypatch, telegram_allowed_users=("12345",))
    assert tg._authorised("12345") is True
    assert tg._authorised("99999") is False
    # A near-miss must not pass — ids are compared as exact strings.
    assert tg._authorised("1234") is False
    assert tg._authorised(" 12345") is False


def test_empty_allow_list_authorises_nobody(monkeypatch):
    """An empty list must mean 'no one', never 'everyone'."""
    _with(monkeypatch, telegram_allowed_users=())
    for candidate in ["1", "12345", "", "admin"]:
        assert tg._authorised(candidate) is False


def test_refuses_to_start_without_a_token(monkeypatch):
    _with(monkeypatch, telegram_token="", telegram_allowed_users=("1",))
    with pytest.raises(RuntimeError, match="TELEGRAM_TOKEN"):
        tg.build_application()


def test_refuses_to_start_with_an_open_allow_list(monkeypatch):
    """Starting wide open would let anyone who finds the bot write memory."""
    _with(monkeypatch, telegram_token="fake:token", telegram_allowed_users=())
    with pytest.raises(RuntimeError, match="TELEGRAM_ALLOWED_USERS"):
        tg.build_application()


def test_allowed_users_parses_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", " 111 , 222,333 ,, ")
    assert Settings().telegram_allowed_users == ("111", "222", "333")


def test_owner_id_is_the_telegram_user_id(monkeypatch):
    """Two Telegram users must get two separate memories.

    This is the seam's whole claim: owner_id has been threaded through
    every query since the first commit, so multi-user needed no schema
    change and no change to run_agent.
    """
    from app.agent.tools import dispatch

    dispatch("upsert_person", {"display_name": "Alice"}, "111")
    dispatch("upsert_person", {"display_name": "Bob"}, "222")

    assert [p["display_name"] for p in dispatch("list_people", {}, "111")["people"]] == ["Alice"]
    assert [p["display_name"] for p in dispatch("list_people", {}, "222")["people"]] == ["Bob"]
    assert dispatch("find_person", {"name": "Alice"}, "222")["match_count"] == 0


# --- voice provenance ------------------------------------------------------


def test_voice_note_keeps_a_link_to_its_recording():
    """A mangled transcript is permanent; the audio must remain reachable.

    Whisper slips on mixed English/Malay/Chinese, and the transcript becomes
    an immutable note. Without audio_path there is no way back to what was
    actually said.
    """
    import asyncio

    from app.agent.loop import run_agent
    from app.agent.tools import dispatch
    from tests.test_loop import ScriptedProvider, _tool_turn
    from app.llm.base import LLMResponse

    provider = ScriptedProvider([
        _tool_turn("c1", "save_note", {"raw_text": "lepak with Wai Keong at the mamak semalam"}),
        LLMResponse(content="Noted."),
    ])
    asyncio.run(
        run_agent(
            "lepak with Wai Keong at the mamak semalam",
            "voice-owner",
            provider=provider,
            source="voice",
            audio_path="audio/123.ogg",
        )
    )

    note = dispatch("search_notes", {"query": "mamak"}, "voice-owner")["notes"][0]
    assert note["source"] == "voice"
    assert note["audio_path"] == "audio/123.ogg"
    # And the words themselves are still verbatim.
    assert note["raw_text"] == "lepak with Wai Keong at the mamak semalam"


def test_a_text_message_is_not_marked_as_voice():
    import asyncio

    from app.agent.loop import run_agent
    from app.agent.tools import dispatch
    from tests.test_loop import ScriptedProvider, _tool_turn
    from app.llm.base import LLMResponse

    provider = ScriptedProvider([
        _tool_turn("c1", "save_note", {"raw_text": "typed this one"}),
        LLMResponse(content="ok"),
    ])
    asyncio.run(run_agent("typed this one", "text-owner", provider=provider))
    note = dispatch("search_notes", {"query": "typed"}, "text-owner")["notes"][0]
    assert note["source"] == "text" and note["audio_path"] is None


def test_model_cannot_forge_voice_provenance():
    """source/audio_path are harness truth, like owner_id."""
    from app.agent.tools import dispatch

    result = dispatch(
        "save_note",
        {"raw_text": "claims to be voice", "_source": "voice", "_audio_path": "/etc/passwd"},
        "forge-owner",
    )
    assert result["ok"]
    note = dispatch("search_notes", {"query": "claims"}, "forge-owner")["notes"][0]
    assert note["source"] == "text"
    assert note["audio_path"] is None


def test_harness_context_reaches_only_save_note():
    from app.agent.tools import dispatch

    pid = dispatch("upsert_person", {"display_name": "Wai Keong"}, "ctx-owner")["person_id"]
    # A non-note tool must not choke on the injected context.
    out = dispatch(
        "get_person_facts", {"person_id": pid}, "ctx-owner",
        context={"_source": "voice", "_audio_path": "audio/x.ogg"},
    )
    assert out["ok"]
