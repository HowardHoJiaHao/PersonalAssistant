"""Brief, reminders, re-derivation and export. No API key needed."""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.agent.tools import dispatch
from app.llm.base import LLMResponse, ToolCall

OWNER = "local"


def _person(name: str, **kw) -> int:
    return dispatch("upsert_person", {"display_name": name, **kw}, OWNER)["person_id"]


def _note(text: str, **kw) -> int:
    return dispatch("save_note", {"raw_text": text, **kw}, OWNER)["note_id"]


def _fact(pid: int, category: str, key: str, value: str, **kw) -> dict:
    return dispatch(
        "save_facts",
        {"facts": [{"person_id": pid, "category": category, "key": key, "value": value, **kw}]},
        OWNER,
    )["saved"][0]


# --- brief -----------------------------------------------------------------


def test_brief_gathers_facts_changes_and_raw_notes():
    pid = _person("Peter", relationship="friend")
    old_note = _note("Peter has been at Maybank for years", person_ids=[pid], event_date="2026-01-10")
    _fact(pid, "work", "employer", "works at Maybank", source_note_id=old_note)

    recent = (date.today() - timedelta(days=5)).isoformat()
    new_note = _note(
        "Peter left Maybank, started at Grab, and mentioned he's been interviewing for a masters",
        person_ids=[pid], event_date=recent,
    )
    _fact(pid, "work", "employer", "works at Grab", source_note_id=new_note)
    _fact(pid, "food", "matcha", "likes matcha a lot", reason="got into it in Kyoto")

    brief = dispatch("get_person_brief", {"person_id": pid}, OWNER)
    assert brief["ok"]

    # Current facts, grouped so the model can present them sensibly.
    assert brief["facts_by_category"]["work"][0]["value"] == "works at Grab"
    assert brief["facts_by_category"]["food"][0]["reason"] == "got into it in Kyoto"

    # Both sides of a recent change: what started AND what ended.
    changed = {(f["value"], f["is_current"]) for f in brief["changed_recently"]}
    assert ("works at Grab", True) in changed
    assert ("works at Maybank", False) in changed

    # Raw notes verbatim — this is what open threads are spotted from.
    assert any("interviewing for a masters" in n["raw_text"] for n in brief["recent_notes"])
    assert brief["days_since_last_seen"] == 5


def test_brief_recent_window_excludes_old_changes():
    pid = _person("Sarah")
    _fact(pid, "work", "employer", "works at Shell", valid_from="2020-03-01")
    brief = dispatch("get_person_brief", {"person_id": pid, "recent_days": 30}, OWNER)
    assert brief["facts_by_category"]["work"][0]["value"] == "works at Shell"
    assert brief["changed_recently"] == []


def test_brief_on_unknown_person_fails_cleanly():
    assert dispatch("get_person_brief", {"person_id": 9999}, OWNER)["ok"] is False


# --- reminders -------------------------------------------------------------


def test_recurring_birthday_surfaces_in_the_window():
    pid = _person("Mei")
    # A birthday years in the past must still be due on its next anniversary.
    soon = date.today() + timedelta(days=10)
    birthday = date(1992, soon.month, soon.day)
    _fact(pid, "milestone", "birthday", "birthday", date_value=birthday.isoformat(), recurring=True)

    upcoming = dispatch("get_reminders", {"within_days": 30}, OWNER)["upcoming"]
    assert [u["person_name"] for u in upcoming] == ["Mei"]
    assert upcoming[0]["days_away"] == 10


def test_recurring_date_already_past_this_year_rolls_to_next():
    pid = _person("Ahmad")
    passed = date.today() - timedelta(days=30)
    _fact(pid, "milestone", "birthday", "birthday",
          date_value=date(1990, passed.month, passed.day).isoformat(), recurring=True)
    # Not due in the next 30 days; due in roughly 11 months.
    assert dispatch("get_reminders", {"within_days": 30}, OWNER)["upcoming"] == []
    far = dispatch("get_reminders", {"within_days": 400}, OWNER)["upcoming"]
    assert far and far[0]["days_away"] > 300


def test_non_recurring_date_does_not_repeat():
    pid = _person("Zoe")
    _fact(pid, "milestone", "wedding", "got married", date_value="2020-06-01", recurring=False)
    assert dispatch("get_reminders", {"within_days": 400}, OWNER)["upcoming"] == []


def test_out_of_touch_ignores_one_off_acquaintances():
    from app.db.session import get_session
    from app.db import repository as repo
    from datetime import datetime, timezone

    seen_often = _person("Old Friend")
    seen_once = _person("Someone From A Party")
    with get_session() as session:
        long_ago = datetime.now(timezone.utc) - timedelta(days=200)
        for pid, count in ((seen_often, 8), (seen_once, 1)):
            person = repo.get_person(session, OWNER, pid)
            person.mention_count, person.last_mentioned = count, long_ago

    stale = dispatch("get_reminders", {"quiet_days": 90}, OWNER)["out_of_touch"]
    names = [p["display_name"] for p in stale]
    assert "Old Friend" in names
    # A single mention should not generate guilt forever.
    assert "Someone From A Party" not in names


# --- re-derivation ---------------------------------------------------------


class _Script:
    """Fake provider replaying one canned extraction per note."""

    def __init__(self, per_note: list[list[LLMResponse]]) -> None:
        self.per_note = per_note
        self.index = -1
        self.step = 0
        self.saw_tools: list[str] = []

    async def complete(self, messages, tools=None, model=None):
        if self.step == 0:
            self.index += 1
        self.saw_tools = [t["function"]["name"] for t in (tools or [])]
        turn = self.per_note[self.index][self.step]
        self.step += 1
        if self.step >= len(self.per_note[self.index]):
            self.step = 0
        return turn


def _call(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id="x", name=name, arguments=args)],
        raw_message={"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]},
    )


def test_rederive_rebuilds_facts_and_never_touches_notes():
    """The design's central claim, tested rather than assumed."""
    from app.agent.rederive import rederive_facts

    pid = _person("Peter", aliases=["peter"])
    n1 = _note("lunch with Peter, he's into matcha, got into it in Kyoto",
               person_ids=[pid], event_date="2026-01-05")
    n2 = _note("Peter started at Grab", person_ids=[pid], event_date="2026-06-02")
    _fact(pid, "food", "matcha", "likes matcha", source_note_id=n1)

    notes_before = dispatch("search_notes", {"query": "", "limit": 100}, OWNER)["notes"]
    stats_before = len(notes_before)

    provider = _Script([
        [_call("save_facts", {"facts": [{"person_id": pid, "category": "food", "key": "matcha",
                                         "value": "loves matcha", "reason": "got into it in Kyoto"}]}),
         LLMResponse(content="done")],
        [_call("save_facts", {"facts": [{"person_id": pid, "category": "work", "key": "employer",
                                         "value": "works at Grab"}]}),
         LLMResponse(content="done")],
    ])
    result = asyncio.run(rederive_facts(OWNER, provider=provider))

    assert result["notes_processed"] == 2
    assert result["facts_wiped"] == 1
    assert result["facts_saved"] == 2

    # Layer 1 is byte-for-byte untouched.
    notes_after = dispatch("search_notes", {"query": "", "limit": 100}, OWNER)["notes"]
    assert len(notes_after) == stats_before
    assert {n["raw_text"] for n in notes_after} == {n["raw_text"] for n in notes_before}

    # Layer 2 was rebuilt, with provenance forced back onto the source note.
    facts = dispatch("get_person_facts", {"person_id": pid}, OWNER)["facts"]
    assert {f["value"] for f in facts} == {"loves matcha", "works at Grab"}
    assert all(f["source_note_id"] in {n1, n2} for f in facts)


def test_rederive_cannot_reach_save_note():
    """Layer 1 must be unreachable from the re-derivation pass."""
    from app.agent.rederive import rederive_facts, ALLOWED_TOOLS

    assert "save_note" not in ALLOWED_TOOLS
    _note("Peter likes durian", event_date="2026-02-02")

    provider = _Script([[_call("save_note", {"raw_text": "injected duplicate"}),
                         LLMResponse(content="done")]])
    asyncio.run(rederive_facts(OWNER, provider=provider))

    # The tool was never offered, and the call was refused.
    assert "save_note" not in provider.saw_tools
    texts = [n["raw_text"] for n in dispatch("search_notes", {"query": "", "limit": 100}, OWNER)["notes"]]
    assert "injected duplicate" not in texts


def test_rederive_replays_notes_in_event_date_order():
    """Out of order, an old job would supersede a new one."""
    from app.agent.rederive import rederive_facts

    pid = _person("Peter")
    _note("Peter joined Grab", person_ids=[pid], event_date="2026-06-02")
    _note("Peter was at Maybank", person_ids=[pid], event_date="2026-01-10")

    seen: list[str] = []

    class Recorder(_Script):
        async def complete(self, messages, tools=None, model=None):
            if self.step == 0:
                seen.append(messages[-1]["content"])
            return await super().complete(messages, tools, model)

    provider = Recorder([[LLMResponse(content="nothing")], [LLMResponse(content="nothing")]])
    asyncio.run(rederive_facts(OWNER, provider=provider))
    assert "Maybank" in seen[0] and "Grab" in seen[1]


# --- export ----------------------------------------------------------------


def test_export_writes_readable_markdown(tmp_path: Path):
    from app.export import export_markdown

    pid = _person("Peter Lim", relationship="friend", how_we_met="badminton")
    note = _note("lunch at Village Park, Peter's into matcha, got into it in Kyoto",
                 person_ids=[pid], event_date="2026-09-23", location="Village Park")
    _fact(pid, "food", "matcha", "likes matcha a lot", reason="got into it in Kyoto", source_note_id=note)
    _fact(pid, "work", "employer", "works at Maybank")
    _fact(pid, "work", "employer", "works at Grab")
    _note("random thought about nobody in particular")

    result = export_markdown(OWNER, tmp_path)
    assert result["ok"]

    written = {p.name for p in tmp_path.glob("*.md")}
    assert "index.md" in written and "unattached-notes.md" in written

    page = next(p for p in tmp_path.glob("peter-lim-*.md")).read_text()
    assert "# Peter Lim" in page
    assert "badminton" in page
    assert "likes matcha a lot" in page and "got into it in Kyoto" in page
    # Superseded facts are preserved as history, not dropped.
    assert "works at Grab" in page and "works at Maybank" in page
    assert "## Previously" in page
    # Notes verbatim — the layer that cannot be regenerated.
    assert "lunch at Village Park, Peter's into matcha" in page
    assert "Village Park" in page

    assert "[Peter Lim](" in (tmp_path / "index.md").read_text()
    assert "random thought" in (tmp_path / "unattached-notes.md").read_text()


def test_future_dated_note_is_a_plan_not_a_last_sighting():
    """Regression from real data: a note about tomorrow gave "last seen -1d"."""
    pid = _person("Waikeong")
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    _note("saw waikeong at the mamak", person_ids=[pid], event_date=yesterday)
    _note("tomorrow waikeong goes to public bank, he wants my photostate ic",
          person_ids=[pid], event_date=tomorrow)

    brief = dispatch("get_person_brief", {"person_id": pid}, OWNER)
    assert brief["days_since_last_seen"] == 1, "a plan must not count as a sighting"
    assert brief["last_seen"] == yesterday
    assert any("public bank" in n["raw_text"] for n in brief["planned"])


def test_planned_notes_surface_in_reminders():
    """A written-down plan is a reminder no fact can express."""
    pid = _person("Waikeong")
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    _note("tomorrow waikeong goes to public bank, remind me in the morning",
          person_ids=[pid], event_date=tomorrow)

    planned = dispatch("get_reminders", {"within_days": 30}, OWNER)["planned"]
    assert len(planned) == 1
    assert planned[0]["days_away"] == 1
    assert planned[0]["people"] == ["Waikeong"]


def test_past_and_distant_notes_are_not_reminders():
    pid = _person("Waikeong")
    _note("went to the bank", person_ids=[pid],
          event_date=(date.today() - timedelta(days=3)).isoformat())
    _note("holiday next year", person_ids=[pid],
          event_date=(date.today() + timedelta(days=200)).isoformat())
    assert dispatch("get_reminders", {"within_days": 30}, OWNER)["planned"] == []
