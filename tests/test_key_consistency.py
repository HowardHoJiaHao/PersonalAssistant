"""The supersession key is the weakest link in the design.

Supersession matches on (person_id, key), and `key` is a free-text string
the model invents. If it drifts, nothing supersedes and BOTH facts read as
current — the app then reports two employers as simultaneously true. It is
silent and looks like normal data, so these tests guard it directly.
"""

from __future__ import annotations

import pytest

from app.agent.tools import dispatch
from app.db.repository import normalise_key

OWNER = "local"


def _person(name: str = "Peter") -> int:
    return dispatch("upsert_person", {"display_name": name}, OWNER)["person_id"]


def _save(pid: int, category: str, key: str, value: str, **kw) -> dict:
    return dispatch(
        "save_facts",
        {"facts": [{"person_id": pid, "category": category, "key": key, "value": value, **kw}]},
        OWNER,
    )["saved"][0]


def _current(pid: int, category: str) -> list[str]:
    facts = dispatch(
        "get_person_facts", {"person_id": pid, "category": category}, OWNER
    )["facts"]
    return [f["value"] for f in facts]


@pytest.mark.parametrize(
    "first_key, second_key",
    [
        ("employer", "workplace"),   # semantic drift, caught by KEY_ALIASES
        ("employer", "company"),
        ("employer", "works at"),
        ("employer", "Employer"),    # case
        ("employer", "employers"),   # spelling drift, caught by fuzzy match
        ("job_title", "jobtitle"),
    ],
)
def test_drifting_keys_still_supersede(first_key, second_key):
    """This is the bug. Two names for one slot must not both stay current."""
    pid = _person()
    _save(pid, "work", first_key, "works at Maybank")
    result = _save(pid, "work", second_key, "works at Grab")

    assert result["action"] == "superseded", (
        f"{second_key!r} did not supersede {first_key!r} — both facts are now "
        "current and the app will report two jobs at once"
    )
    assert _current(pid, "work") == ["works at Grab"]

    past = dispatch(
        "get_person_facts", {"person_id": pid, "category": "work", "include_past": True},
        OWNER,
    )["facts"]
    closed = [f for f in past if f["value"] == "works at Maybank"]
    assert len(closed) == 1 and closed[0]["valid_to"] is not None


@pytest.mark.parametrize(
    "key_a, key_b",
    [
        ("matcha", "mocha"),     # 73 — different drinks, must NOT merge
        ("coffee", "toffee"),    # 83 — closest false pair measured
        ("sister", "mister"),
        ("beer", "deer"),
        ("matcha", "matcha_latte"),
    ],
)
def test_genuinely_different_keys_are_not_merged(key_a, key_b):
    """The threshold must not over-merge. A wrong merge loses a real fact."""
    pid = _person()
    _save(pid, "food", key_a, f"likes {key_a}")
    result = _save(pid, "food", key_b, f"likes {key_b}")

    assert result["action"] == "created", f"{key_b!r} was wrongly merged into {key_a!r}"
    assert sorted(_current(pid, "food")) == sorted([f"likes {key_a}", f"likes {key_b}"])


def test_collapse_is_reported_to_the_model():
    pid = _person()
    _save(pid, "work", "employer", "works at Maybank")
    result = _save(pid, "work", "workplace", "works at Grab")
    assert "note_to_model" in result
    assert "superseded" in result["note_to_model"]


def test_new_sibling_key_is_flagged_so_the_model_can_self_correct():
    pid = _person()
    _save(pid, "food", "matcha", "likes matcha")
    result = _save(pid, "food", "durian", "loves durian")
    assert result["action"] == "created"
    assert result["sibling_keys"] == ["matcha"]
    assert "correct_fact" in result["note_to_model"]


def test_keys_are_scoped_per_person_and_per_category():
    """A resolution must never reach across people or categories."""
    a, b = _person("Peter"), _person("Sarah")
    _save(a, "work", "employer", "works at Maybank")
    # Same key, different person -> a brand new fact, not a supersession.
    assert _save(b, "work", "employers", "works at Grab")["action"] == "created"
    assert _current(a, "work") == ["works at Maybank"]
    assert _current(b, "work") == ["works at Grab"]

    # Same key text, different category -> also independent.
    assert _save(a, "hobby", "employers", "runs a side business")["action"] == "created"
    assert _current(a, "work") == ["works at Maybank"]


def test_normalise_key_is_stable():
    for raw in ["Employer", "  employer ", "EMPLOYER", "employer"]:
        assert normalise_key(raw) == "employer"
    for raw in ["workplace", "work", "job", "company", "works at", "WORKS_AT"]:
        assert normalise_key(raw) == "employer", raw
    assert normalise_key("dob") == "birthday"
    assert normalise_key("matcha") == "matcha"
