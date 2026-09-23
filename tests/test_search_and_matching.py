"""Full-text search and person matching.

Both replaced TODO stubs, and both had to keep their signatures. The
matching tests exist mainly to prove fuzzy recall did NOT weaken the
"never guess which person" guard rail.
"""

from __future__ import annotations

from app.agent.tools import dispatch
from app.db.session import engine
from app.db.repository import _fts_available
from app.db.session import get_session

OWNER = "local"


def _note(text: str, **kw) -> int:
    return dispatch("save_note", {"raw_text": text, **kw}, OWNER)["note_id"]


def _texts(query: str, **kw) -> list[str]:
    return [n["raw_text"] for n in dispatch("search_notes", {"query": query, **kw}, OWNER)["notes"]]


def test_fts_index_is_built():
    with get_session() as session:
        assert _fts_available(session), "FTS5 should be available in this build"


def test_search_stems_and_matches_across_words():
    _note("Sarah is training for the Penang bridge run in December")
    _note("Peter said he's thinking about applying for jobs in Singapore")
    _note("had lunch with Peter at Village Park, he's into matcha, got into it in Kyoto")

    # Stemming: the note says "run", the query says "running".
    assert any("bridge run" in t for t in _texts("running"))
    # Stemming the other way: note says "applying", query says "applied".
    assert any("applying for jobs" in t for t in _texts("applied"))
    # Multi-word, non-adjacent — LIKE could never do this.
    assert any("Kyoto" in t for t in _texts("matcha Kyoto"))


def test_search_handles_code_switched_notes():
    """Notes here mix English, Malay and Chinese, often in one sentence."""
    _note("makan with Ahmad at the mamak, he ordered teh tarik again")
    assert _texts("mamak")
    assert _texts("teh tarik")


def test_search_survives_characters_that_would_break_a_raw_match_expression():
    """Quotes and operators must not reach FTS5 as syntax."""
    _note("Peter said \"I'm done with Maybank\" and laughed")
    for query in ['"', "AND", "NOT OR", "*", "()", "'"]:
        dispatch("search_notes", {"query": query}, OWNER)  # must not raise
    assert _texts("Maybank")


def test_search_is_scoped_by_owner_and_person():
    mine = _note("Peter loves durian")
    dispatch("save_note", {"raw_text": "Peter loves durian"}, "someone-else")
    assert len(_texts("durian")) == 1

    pid = dispatch("upsert_person", {"display_name": "Peter"}, OWNER)["person_id"]
    dispatch("save_note", {"raw_text": "Ahmad loves durian too", "person_ids": []}, OWNER)
    dispatch("save_note", {"raw_text": "Peter tried durian ice cream", "person_ids": [pid]}, OWNER)
    scoped = dispatch("search_notes", {"query": "durian", "person_id": pid}, OWNER)["notes"]
    assert all(pid in n["person_ids"] for n in scoped)
    assert len(scoped) == 1


# --- person matching -------------------------------------------------------


def test_exact_match_does_not_become_ambiguous_because_of_a_similar_name():
    """The key risk of adding fuzzy matching: manufactured ambiguity."""
    dispatch("upsert_person", {"display_name": "Peter", "aliases": ["peter"]}, OWNER)
    dispatch("upsert_person", {"display_name": "Pieter", "aliases": ["pieter"]}, OWNER)

    found = dispatch("find_person", {"name": "peter"}, OWNER)
    assert found["match_count"] == 1
    assert found["ambiguous"] is False
    # The near-miss is still reported, just not as a match.
    assert [p["display_name"] for p in found["similar_names"]] == ["Pieter"]


def test_two_real_matches_are_still_flagged_ambiguous():
    """The original hard rule must survive the rewrite."""
    dispatch("upsert_person", {"display_name": "Peter", "aliases": ["peter"], "disambiguator": "badminton"}, OWNER)
    dispatch("upsert_person", {"display_name": "Peter Tan", "aliases": ["peter"], "disambiguator": "work"}, OWNER)

    found = dispatch("find_person", {"name": "peter"}, OWNER)
    assert found["match_count"] == 2
    assert found["ambiguous"] is True
    assert "Do NOT guess" in found["hint"]


def test_fuzzy_only_promotes_when_nothing_matched_literally():
    """Typos should still find someone, which is the whole point."""
    dispatch("upsert_person", {"display_name": "Siti Nurhaliza", "aliases": ["siti"]}, OWNER)
    found = dispatch("find_person", {"name": "Sitti"}, OWNER)
    assert [p["display_name"] for p in found["matches"]] == ["Siti Nurhaliza"]


def test_context_orders_matches_but_never_resolves_them():
    badminton = dispatch(
        "upsert_person", {"display_name": "Peter", "aliases": ["peter"], "disambiguator": "badminton, Bangsar"}, OWNER
    )["person_id"]
    work = dispatch(
        "upsert_person", {"display_name": "Peter Tan", "aliases": ["peter"], "disambiguator": "work"}, OWNER
    )["person_id"]

    found = dispatch("find_person", {"name": "peter", "context": "badminton in Bangsar"}, OWNER)
    # Context ranks the right Peter first...
    assert found["matches"][0]["person_id"] == badminton
    # ...but must NOT collapse the choice.
    assert found["match_count"] == 2 and found["ambiguous"] is True


def test_upsert_never_reuses_a_row_on_a_fuzzy_name():
    """Merging two different people on a near-miss is unrecoverable."""
    a = dispatch("upsert_person", {"display_name": "Siti"}, OWNER)["person_id"]
    b = dispatch("upsert_person", {"display_name": "Sitti"}, OWNER)["person_id"]
    assert a != b


def test_index_is_backfilled_for_notes_that_predate_it(tmp_path):
    """Regression: notes written before the FTS index existed must be indexed.

    This shipped broken. The backfill was gated on COUNT(*) FROM notes_fts,
    but that is an external-content table — COUNT reads the notes table and
    reports a healthy number even when the index is empty. The rebuild was
    skipped, every search fell through to LIKE, and the only visible symptom
    was that stemming quietly stopped working.
    """
    import sqlite3

    from app.db import session as db_session

    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE notes (id INTEGER PRIMARY KEY, owner_id VARCHAR(64),
          raw_text TEXT, source VARCHAR(20) DEFAULT 'text', audio_path VARCHAR(500),
          person_ids JSON, event_date DATE, location VARCHAR(200),
          activity VARCHAR(200), created_at DATETIME);
        INSERT INTO notes (owner_id, raw_text, person_ids)
          VALUES ('local', 'waikeong is super friendly and he talk very fast', '[]');
        """
    )
    con.commit()
    con.close()

    try:
        db_session.configure_for_testing(f"sqlite:///{db}")
        with get_session() as session:
            assert _fts_available(session)
            # "talking" only finds "talk" through the stemmed index; LIKE
            # cannot do this, so a hit proves the backfill actually ran.
            hits = dispatch("search_notes", {"query": "talking"}, OWNER)
        assert hits["notes"], "pre-existing note was never indexed"
        assert "talk very fast" in hits["notes"][0]["raw_text"]
    finally:
        db_session.engine.dispose()


def test_stemming_beats_like_on_word_endings():
    """Guards the same failure from the other side: if FTS silently stops
    working, these queries fall back to LIKE and return nothing."""
    _note("waikeong is super friendly and he talk very fast and encourages people")
    # Each query differs from the stored word only by its ending.
    assert _texts("talking"), "stemming inactive — FTS has fallen back to LIKE"
    assert _texts("encouraging")
