"""The owner is a person too.

Modelled as an ordinary Person row with is_self, so facts, relations,
notes and search work on the user with no separate machinery. The risk
this creates is mis-routing: "I'm allergic to prawns" landing on whoever
was mentioned last is both a lost fact and a false one, and it gets acted
on at a dinner table.
"""

from __future__ import annotations

from app.agent.tools import dispatch

OWNER = "local"


def _me(**kw) -> dict:
    return dispatch("remember_about_me", kw, OWNER)


def test_facts_about_me_are_stored_and_readable():
    _me(display_name="Howard",
        facts=[{"category": "health", "key": "prawns", "value": "allergic to prawns"},
               {"category": "work", "key": "employer", "value": "works at Maybank"}])

    me = dispatch("get_about_me", {}, OWNER)
    assert me["known"] is True
    assert me["me"]["display_name"] == "Howard"
    assert me["me"]["is_self"] is True
    assert {f["value"] for f in me["facts"]} == {
        "allergic to prawns", "works at Maybank"
    }


def test_nothing_is_invented_when_the_user_is_unknown():
    out = dispatch("get_about_me", {}, OWNER)
    assert out["known"] is False
    assert "invent" in out["hint"]


def test_first_person_words_resolve_to_the_owner():
    _me(display_name="Howard", facts=[])
    for word in ["me", "Me", "myself", "I", "my"]:
        found = dispatch("find_person", {"name": word}, OWNER)
        assert found["match_count"] == 1, word
        assert found["matches"][0]["is_self"] is True, word


def test_self_row_is_reused_not_duplicated():
    _me(display_name="Howard", facts=[{"category": "food", "key": "durian", "value": "loves durian"}])
    _me(facts=[{"category": "food", "key": "coriander", "value": "hates coriander"}])
    # upsert_person("me") must find the same row rather than make a second.
    again = dispatch("upsert_person", {"display_name": "me"}, OWNER)
    assert again["is_self"] is True

    with_self = dispatch("find_person", {"name": "me"}, OWNER)
    assert with_self["match_count"] == 1
    assert len(dispatch("get_about_me", {}, OWNER)["facts"]) == 2


def test_my_facts_do_not_leak_into_who_do_i_know():
    _me(display_name="Howard", facts=[{"category": "food", "key": "durian", "value": "loves durian"}])
    dispatch("upsert_person", {"display_name": "Wai Keong"}, OWNER)

    names = [p["display_name"] for p in dispatch("list_people", {}, OWNER)["people"]]
    assert names == ["Wai Keong"], "'who do I know' is about other people"


def test_my_facts_are_still_findable_by_search():
    """Excluded from the roster, but not hidden from questions."""
    _me(display_name="Howard", facts=[{"category": "food", "key": "durian", "value": "loves durian"}])
    dispatch("upsert_person", {"display_name": "Wai Keong"}, OWNER)
    dispatch("save_facts", {"facts": [{"person_id": 2, "category": "food",
                                       "key": "durian", "value": "hates durian"}]}, OWNER)

    hits = {r["person_name"]: r["value"] for r in
            dispatch("search_facts", {"query": "durian"}, OWNER)["results"]}
    assert hits == {"Howard": "loves durian", "Wai Keong": "hates durian"}


def test_my_facts_supersede_like_anyone_elses():
    _me(display_name="Howard", facts=[{"category": "work", "key": "employer", "value": "works at Maybank"}])
    _me(facts=[{"category": "work", "key": "workplace", "value": "works at Grab"}])

    current = [f["value"] for f in dispatch("get_about_me", {}, OWNER)["facts"]]
    assert current == ["works at Grab"], "key drift must supersede for the user too"


def test_relations_between_me_and_others():
    """'my sister is Mei' — the user is just another node in the graph."""
    me = _me(display_name="Howard", facts=[])["me"]["person_id"]
    mei = dispatch("upsert_person", {"display_name": "Mei"}, OWNER)["person_id"]
    dispatch("save_relation", {"from_person": me, "to_person": mei, "type": "sibling"}, OWNER)

    mine = dispatch("get_about_me", {}, OWNER)["relations"]
    assert mine[0]["relation"] == "sibling" and mine[0]["other_name"] == "Mei"
    # And it reads correctly from her side.
    hers = dispatch("get_relations", {"person_id": mei}, OWNER)["relations"]
    assert hers[0]["other_name"] == "Howard" and hers[0]["relation"] == "sibling"


def test_stats_counts_people_i_know_not_me():
    from app.db.session import get_session
    from app.db import repository as repo

    _me(display_name="Howard", facts=[])
    dispatch("upsert_person", {"display_name": "Wai Keong"}, OWNER)
    with get_session() as session:
        assert repo.stats(session, OWNER)["people"] == 1


def test_self_is_scoped_per_owner():
    _me(display_name="Howard", facts=[{"category": "health", "key": "prawns", "value": "allergic to prawns"}])
    assert dispatch("get_about_me", {}, "someone-else")["known"] is False
