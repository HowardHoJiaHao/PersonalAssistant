"""Backend scenario test — repository + tool dispatcher, no LLM, no API key.

Walks one realistic story end to end and prints what it sees, so the
output doubles as a readable demo of the two memory layers.
"""

from __future__ import annotations

from app.agent.tools import dispatch

OWNER = "local"


def show(title: str, payload: object) -> None:
    print(f"\n--- {title} ---")
    print(payload)


def test_full_backend_scenario(capsys):
    with capsys.disabled():
        _scenario()


def _scenario() -> None:
    # ------------------------------------------------------------------
    # 1. the Peter / matcha / Village Park note
    # ------------------------------------------------------------------
    raw = (
        "had lunch with Peter at Village Park today, he's really into matcha now, "
        "got into it in Kyoto last year, he was late again lol"
    )
    note = dispatch(
        "save_note",
        {
            "raw_text": raw,
            "event_date": "2026-09-23",
            "location": "Village Park",
            "activity": "lunch",
        },
        OWNER,
    )
    assert note["ok"]
    # Layer 1 is verbatim: the stored text is character-for-character mine.
    assert note["raw_text"] == raw
    show("1. note saved (layer 1, verbatim)", note["raw_text"])

    peter = dispatch(
        "upsert_person",
        {
            "display_name": "Peter",
            "aliases": ["peter"],
            "relationship": "friend",
            "disambiguator": "badminton, Bangsar",
        },
        OWNER,
    )
    assert peter["ok"]
    peter_id = peter["person_id"]

    saved = dispatch(
        "save_facts",
        {
            "facts": [
                {
                    "person_id": peter_id,
                    "category": "food",
                    "key": "matcha",
                    "value": "likes matcha a lot",
                    "reason": "got into it in Kyoto",
                    "source_note_id": note["note_id"],
                }
            ]
        },
        OWNER,
    )
    assert saved["ok"] and saved["saved"][0]["action"] == "created"
    matcha_fact_id = saved["saved"][0]["fact"]["fact_id"]
    # The note mentioned lunch, the restaurant and him being late; none of
    # those are durable, so exactly one fact came out of it.
    current = dispatch("get_person_facts", {"person_id": peter_id}, OWNER)
    assert len(current["facts"]) == 1
    show("1. one note -> one fact", current["facts"])

    # Fact dates follow the note's event_date, not the wall clock.
    assert saved["saved"][0]["fact"]["valid_from"] == "2026-09-23"

    # ------------------------------------------------------------------
    # 2. Peter leaves Maybank for Grab -> supersession, not overwrite
    # ------------------------------------------------------------------
    maybank_note = dispatch(
        "save_note",
        {
            "raw_text": "Peter has been at Maybank for three years now",
            "person_ids": [peter_id],
            "event_date": "2026-01-10",
        },
        OWNER,
    )
    dispatch(
        "save_facts",
        {
            "facts": [
                {
                    "person_id": peter_id,
                    "category": "work",
                    "key": "employer",
                    "value": "works at Maybank",
                    "source_note_id": maybank_note["note_id"],
                }
            ]
        },
        OWNER,
    )

    grab_note = dispatch(
        "save_note",
        {
            "raw_text": "caught up with Peter, he left Maybank and started at Grab on Tuesday",
            "person_ids": [peter_id],
            "event_date": "2026-06-02",  # Tuesday, though written up later
        },
        OWNER,
    )
    grab = dispatch(
        "save_facts",
        {
            "facts": [
                {
                    "person_id": peter_id,
                    "category": "work",
                    "key": "employer",
                    "value": "works at Grab",
                    "source_note_id": grab_note["note_id"],
                }
            ]
        },
        OWNER,
    )
    assert grab["saved"][0]["action"] == "superseded"

    current_work = dispatch(
        "get_person_facts", {"person_id": peter_id, "category": "work"}, OWNER
    )["facts"]
    assert [f["value"] for f in current_work] == ["works at Grab"]
    # "when did he start at Grab?" -> Tuesday, the event date, not Thursday.
    assert current_work[0]["valid_from"] == "2026-06-02"
    show("2. current work facts (Grab only)", current_work)

    past_work = dispatch(
        "get_person_facts",
        {"person_id": peter_id, "category": "work", "include_past": True},
        OWNER,
    )["facts"]
    maybank = [f for f in past_work if f["value"] == "works at Maybank"]
    assert len(maybank) == 1, "Maybank must survive the move"
    assert maybank[0]["valid_to"] == "2026-06-02", "closed with an end date"
    assert maybank[0]["is_current"] is False
    show("2. with include_past (Maybank kept, ended)", past_work)

    # Re-stating the same employer is a no-op, not a duplicate row.
    again = dispatch(
        "save_facts",
        {
            "facts": [
                {
                    "person_id": peter_id,
                    "category": "work",
                    "key": "employer",
                    "value": "works at Grab",
                }
            ]
        },
        OWNER,
    )
    assert again["saved"][0]["action"] == "unchanged"

    # ------------------------------------------------------------------
    # 3. provenance — when / where / why live in the note
    # ------------------------------------------------------------------
    prov = dispatch("get_fact_provenance", {"fact_id": matcha_fact_id}, OWNER)
    assert prov["ok"]
    assert prov["note"]["event_date"] == "2026-09-23"
    assert prov["note"]["location"] == "Village Park"
    assert "got into it in Kyoto" in prov["note"]["raw_text"]
    assert prov["fact"]["reason"] == "got into it in Kyoto"
    show("3. provenance: date", prov["note"]["event_date"])
    show("3. provenance: location", prov["note"]["location"])
    show("3. provenance: my original wording", prov["note"]["raw_text"])

    # ------------------------------------------------------------------
    # 4. a second Peter -> find_person must return BOTH, flagged ambiguous
    # ------------------------------------------------------------------
    peter2 = dispatch(
        "upsert_person",
        {
            "display_name": "Peter Tan",
            "aliases": ["peter"],
            "relationship": "colleague",
            "disambiguator": "work, the one from the Grab team",
        },
        OWNER,
    )
    found = dispatch("find_person", {"name": "peter"}, OWNER)
    assert found["match_count"] == 2, found
    assert found["ambiguous"] is True
    assert "ASK the user" in found["hint"]
    show("4. find_person('peter')", found)

    # ------------------------------------------------------------------
    # 5. cross-person search
    # ------------------------------------------------------------------
    dispatch(
        "save_facts",
        {
            "facts": [
                {
                    "person_id": peter2["person_id"],
                    "category": "food",
                    "key": "matcha",
                    "value": "also drinks matcha every morning",
                }
            ]
        },
        OWNER,
    )
    fact_hits = dispatch("search_facts", {"query": "matcha"}, OWNER)["results"]
    assert {f["person_name"] for f in fact_hits} == {"Peter", "Peter Tan"}
    show("5. search_facts('matcha') across people", fact_hits)

    note_hits = dispatch("search_notes", {"query": "Village Park"}, OWNER)["notes"]
    assert len(note_hits) == 1 and "Kyoto" in note_hits[0]["raw_text"]
    show("5. search_notes('Village Park')", note_hits)

    # ------------------------------------------------------------------
    # 6. relations, then a correction
    # ------------------------------------------------------------------
    rel = dispatch(
        "save_relation",
        {
            "from_person": peter_id,
            "to_person": peter2["person_id"],
            "type": "colleague",
            "valid_from": "2026-06-02",
        },
        OWNER,
    )
    assert rel["ok"]
    relations = dispatch("get_relations", {"person_id": peter_id}, OWNER)["relations"]
    assert len(relations) == 1
    assert relations[0]["from_name"] == "Peter"
    assert relations[0]["to_name"] == "Peter Tan"
    show("6. get_relations", relations)

    fixed = dispatch(
        "correct_fact",
        {"fact_id": matcha_fact_id, "new_value": "loves matcha, drinks it daily"},
        OWNER,
    )
    assert fixed["ok"] and fixed["fact"]["value"] == "loves matcha, drinks it daily"
    show("6. correct_fact", fixed["fact"])

    moved = dispatch(
        "correct_fact",
        {"fact_id": matcha_fact_id, "move_to_person_id": peter2["person_id"]},
        OWNER,
    )
    assert moved["ok"]
    assert (
        dispatch("get_person_facts", {"person_id": peter2["person_id"]}, OWNER)[
            "facts"
        ].__len__()
        == 2
    )
    # Move it back so the delete case runs against the original owner.
    dispatch("correct_fact", {"fact_id": matcha_fact_id, "move_to_person_id": peter_id}, OWNER)
    deleted = dispatch("correct_fact", {"fact_id": matcha_fact_id, "delete": True}, OWNER)
    assert deleted["action"] == "deleted"
    # The note it came from is untouched — facts are disposable, notes are not.
    assert dispatch("search_notes", {"query": "Kyoto"}, OWNER)["notes"]
    show("6. fact deleted, source note survives", "note still searchable")


def test_owner_id_is_never_taken_from_model_args():
    """The tenant boundary must not be reachable from tool arguments."""
    dispatch("upsert_person", {"display_name": "Alice"}, "owner-a")

    # A model that invents owner_id must not be able to read another tenant.
    leaked = dispatch("list_people", {"owner_id": "owner-a"}, "owner-b")
    assert leaked["people"] == []

    mine = dispatch("list_people", {}, "owner-a")
    assert [p["display_name"] for p in mine["people"]] == ["Alice"]

    # No tool schema may even declare owner_id as a parameter.
    from app.agent.tools import TOOL_SCHEMAS

    for schema in TOOL_SCHEMAS:
        params = schema["function"]["parameters"].get("properties", {})
        assert "owner_id" not in params, schema["function"]["name"]


def test_no_delete_person_tool():
    """A tool that doesn't exist can't be misused."""
    from app.agent.tools import TOOL_IMPLS

    assert "delete_person" not in TOOL_IMPLS
    assert dispatch("delete_person", {"person_id": 1}, OWNER)["ok"] is False


def test_merge_people_moves_everything():
    a = dispatch("upsert_person", {"display_name": "Sarah Lim"}, OWNER)
    b = dispatch("upsert_person", {"display_name": "Sarah L"}, OWNER)
    note = dispatch(
        "save_note", {"raw_text": "Sarah started running", "person_ids": [b["person_id"]]}, OWNER
    )
    dispatch(
        "save_facts",
        {
            "facts": [
                {
                    "person_id": b["person_id"],
                    "category": "hobby",
                    "key": "running",
                    "value": "runs most mornings",
                    "source_note_id": note["note_id"],
                }
            ]
        },
        OWNER,
    )
    result = dispatch(
        "merge_people", {"keep_id": a["person_id"], "merge_id": b["person_id"]}, OWNER
    )
    assert result["ok"] and result["moved_facts"] == 1 and result["moved_notes"] == 1
    kept = dispatch("get_person_facts", {"person_id": a["person_id"]}, OWNER)
    assert [f["value"] for f in kept["facts"]] == ["runs most mornings"]
    assert dispatch("find_person", {"name": "sarah l"}, OWNER)["match_count"] == 1


def test_facts_are_rederivable_from_notes():
    """Wiping layer 2 must lose nothing that layer 1 still holds."""
    from app.db import repository as repo
    from app.db.session import get_session

    p = dispatch("upsert_person", {"display_name": "Mei"}, OWNER)
    note = dispatch(
        "save_note",
        {"raw_text": "Mei hates coriander, always picks it out", "person_ids": [p["person_id"]]},
        OWNER,
    )
    dispatch(
        "save_facts",
        {
            "facts": [
                {
                    "person_id": p["person_id"],
                    "category": "dislike",
                    "key": "coriander",
                    # Negation preserved exactly — never stored as a preference.
                    "value": "hates coriander",
                    "source_note_id": note["note_id"],
                }
            ]
        },
        OWNER,
    )
    with get_session() as session:
        wiped = repo.delete_facts_for_owner(session, OWNER)
    assert wiped >= 1
    assert dispatch("get_person_facts", {"person_id": p["person_id"]}, OWNER)["facts"] == []
    # The words survive, so extraction can simply run again.
    surviving = dispatch("search_notes", {"query": "coriander"}, OWNER)["notes"]
    assert surviving[0]["raw_text"] == "Mei hates coriander, always picks it out"
