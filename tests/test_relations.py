"""Relation direction.

Stored rows are directional: from --type--> to reads "to is the <type> of
from". Handing that row to the other person unchanged inverts the meaning,
and the result looks like perfectly ordinary data — Almond's profile
claiming his child is Wai Keong. These tests pin both ends.
"""

from __future__ import annotations

import pytest

from app.agent.tools import dispatch
from app.db.models import INVERSE_RELATIONS, RELATION_TYPES

OWNER = "local"


def _pair():
    w = dispatch("upsert_person", {"display_name": "Wai Keong"}, OWNER)["person_id"]
    a = dispatch("upsert_person", {"display_name": "Almond"}, OWNER)["person_id"]
    return w, a


def test_parent_child_reads_correctly_from_both_sides():
    """'Wai Keong's son is Almond' must never read as the reverse."""
    w, a = _pair()
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": "child"}, OWNER)

    from_dad = dispatch("get_relations", {"person_id": w}, OWNER)["relations"][0]
    assert from_dad["relation"] == "child"
    assert from_dad["other_name"] == "Almond"
    assert from_dad["phrase"] == "Almond is Wai Keong's child"

    from_kid = dispatch("get_relations", {"person_id": a}, OWNER)["relations"][0]
    # The same row, inverted — NOT "Almond's child is Wai Keong".
    assert from_kid["relation"] == "parent"
    assert from_kid["other_name"] == "Wai Keong"
    assert from_kid["phrase"] == "Wai Keong is Almond's parent"


def test_one_row_serves_both_directions():
    """Storing the reciprocal separately would let the two disagree."""
    w, a = _pair()
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": "child"}, OWNER)
    assert len(dispatch("get_relations", {"person_id": w}, OWNER)["relations"]) == 1
    assert len(dispatch("get_relations", {"person_id": a}, OWNER)["relations"]) == 1


@pytest.mark.parametrize("relation", ["spouse", "sibling", "colleague", "friend"])
def test_symmetric_relations_read_the_same_both_ways(relation):
    w, a = _pair()
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": relation}, OWNER)
    assert dispatch("get_relations", {"person_id": w}, OWNER)["relations"][0]["relation"] == relation
    assert dispatch("get_relations", {"person_id": a}, OWNER)["relations"][0]["relation"] == relation


def test_every_relation_type_has_an_inverse():
    """A missing inverse would silently show the raw type on the wrong side."""
    for relation in RELATION_TYPES:
        assert relation in INVERSE_RELATIONS, relation
    # And inverting twice returns the original for symmetric types.
    for relation in ("spouse", "sibling", "colleague", "friend"):
        assert INVERSE_RELATIONS[INVERSE_RELATIONS[relation]] == relation
    assert INVERSE_RELATIONS["parent"] == "child"
    assert INVERSE_RELATIONS["child"] == "parent"


def test_relation_appears_in_the_brief_from_the_right_side():
    w, a = _pair()
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": "child"}, OWNER)
    brief = dispatch("get_person_brief", {"person_id": a}, OWNER)
    assert brief["relations"][0]["relation"] == "parent"
    assert brief["relations"][0]["other_name"] == "Wai Keong"


def test_relation_needs_both_people_to_exist():
    w, _ = _pair()
    assert dispatch("save_relation", {"from_person": w, "to_person": 9999, "type": "child"}, OWNER)["ok"] is False


def test_relations_do_not_leak_across_owners():
    w, a = _pair()
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": "child"}, OWNER)
    assert dispatch("get_relations", {"person_id": w}, "other-owner")["relations"] == []


def test_ending_a_relation_keeps_it_as_history():
    w, a = _pair()
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": "colleague"}, OWNER)
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": "colleague",
                               "valid_to": "2026-06-01"}, OWNER)
    assert dispatch("get_relations", {"person_id": w}, OWNER)["relations"] == []
    past = dispatch("get_relations", {"person_id": w, "include_past": True}, OWNER)["relations"]
    assert past[0]["valid_to"] == "2026-06-01" and past[0]["is_current"] is False


def test_the_note_still_holds_the_word_the_user_used():
    """'son' is not a relation type — but it is never lost.

    RELATION_TYPES has `child`, so son/daughter is flattened. The raw note
    keeps the user's actual word, which is exactly what layer 1 is for.
    """
    w, a = _pair()
    dispatch("save_note", {"raw_text": "waikeong son is almond", "person_ids": [w, a]}, OWNER)
    dispatch("save_relation", {"from_person": w, "to_person": a, "type": "child"}, OWNER)

    assert dispatch("get_relations", {"person_id": w}, OWNER)["relations"][0]["relation"] == "child"
    assert "son" in dispatch("search_notes", {"query": "son"}, OWNER)["notes"][0]["raw_text"]
