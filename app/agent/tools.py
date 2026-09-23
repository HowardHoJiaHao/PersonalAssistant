"""Tool schemas and the dispatcher.

Two rules shape this file:

1. `owner_id` is NEVER a tool parameter. It is injected by `dispatch`
   from the harness. If the model could name whose data to read, a
   confused or manipulated model could step across the tenant boundary.
   Nothing below may accept it from the model's arguments.

2. There is no `delete_person` tool. A tool that doesn't exist can't be
   misused, and person deletion has no benign use here — merge_people
   covers the real case (duplicates) without destroying anything.

Tools never build SQL; every one of them calls into repository.py.
"""

from __future__ import annotations

from typing import Any, Callable

from app.db import repository as repo
from app.db.models import FACT_CATEGORIES, RELATION_TYPES
from app.db.session import get_session

# ---------------------------------------------------------------------------
# schemas — OpenAI-compatible function definitions
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": (
                "Save the user's message VERBATIM as an immutable note. Call this "
                "FIRST whenever the message contains any information about a person. "
                "Never paraphrase, summarise or clean up raw_text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_text": {
                        "type": "string",
                        "description": "The user's exact words, unedited.",
                    },
                    "person_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Ids of people the note is about, if known yet.",
                    },
                    "event_date": {
                        "type": "string",
                        "description": (
                            "YYYY-MM-DD when the event happened (not when it was "
                            "written). Resolve 'today'/'yesterday'/'last Tuesday' "
                            "against today's date."
                        ),
                    },
                    "location": {"type": "string"},
                    "activity": {"type": "string"},
                },
                "required": ["raw_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upsert_person",
            "description": (
                "Create a person, or update one by person_id. Only pass person_id "
                "when find_person has confirmed exactly who this is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "display_name": {"type": "string"},
                    "person_id": {
                        "type": "integer",
                        "description": "Update this existing person instead of creating one.",
                    },
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "relationship": {
                        "type": "string",
                        "description": "friend | family | colleague",
                    },
                    "how_we_met": {"type": "string"},
                    "disambiguator": {
                        "type": "string",
                        "description": "Short tag like 'badminton, Bangsar' used to tell two same-named people apart.",
                    },
                },
                "required": ["display_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_facts",
            "description": (
                "Save one or more DURABLE facts about people. Most messages yield "
                "zero or one fact — that is correct, do not pad. Always call "
                "find_person first. A new value for an existing key supersedes the "
                "old one automatically; the old value stays answerable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "person_id": {"type": "integer"},
                                "category": {
                                    "type": "string",
                                    "enum": list(FACT_CATEGORIES),
                                },
                                "key": {
                                    "type": "string",
                                    "description": (
                                        "Stable slot for this kind of fact, e.g. "
                                        "'employer', 'matcha', 'coriander'. Reuse the "
                                        "same key when a value changes so supersession works."
                                    ),
                                },
                                "value": {
                                    "type": "string",
                                    "description": (
                                        "The claim. Preserve negations exactly: "
                                        "'hates coriander' is not a coriander preference."
                                    ),
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "The stated why, e.g. 'got into it in Kyoto'.",
                                },
                                "confidence": {"type": "number"},
                                "source_note_id": {
                                    "type": "integer",
                                    "description": "Id from save_note. Required for when/why questions to work later.",
                                },
                                "date_value": {
                                    "type": "string",
                                    "description": (
                                        "YYYY-MM-DD the fact is ABOUT — a birthday, "
                                        "anniversary or start date. Not when you "
                                        "learned it. Enables reminders."
                                    ),
                                },
                                "recurring": {
                                    "type": "boolean",
                                    "description": "True for dates that come round yearly (birthdays, anniversaries).",
                                },
                            },
                            "required": ["person_id", "category", "key", "value"],
                        },
                    }
                },
                "required": ["facts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_relation",
            "description": (
                "Record a relationship between two existing people. DIRECTION "
                "MATTERS: from_person --type--> to_person reads as 'to_person is "
                "the <type> of from_person'. So \"Wai Keong's son is Almond\" is "
                "from_person=Wai Keong, type=child, to_person=Almond. Store it "
                "once; the reverse direction is derived automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_person": {"type": "integer"},
                    "to_person": {"type": "integer"},
                    "type": {"type": "string", "enum": list(RELATION_TYPES)},
                    "valid_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "valid_to": {
                        "type": "string",
                        "description": "YYYY-MM-DD; set to end a relation rather than deleting it.",
                    },
                },
                "required": ["from_person", "to_person", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_fact",
            "description": (
                "Fix a WRONG extraction: change its value, move it to another "
                "person, or delete it. Do not use this when the world changed — "
                "save_facts already supersedes and keeps the history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_id": {"type": "integer"},
                    "new_value": {"type": "string"},
                    "new_reason": {"type": "string"},
                    "move_to_person_id": {"type": "integer"},
                    "delete": {"type": "boolean"},
                },
                "required": ["fact_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_people",
            "description": (
                "Merge a duplicate person into the one to keep. All notes, facts "
                "and relations move across. Only after the user confirms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keep_id": {"type": "integer"},
                    "merge_id": {"type": "integer"},
                },
                "required": ["keep_id", "merge_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_person",
            "description": (
                "Look up people by name or alias. Returns ALL plausible matches. "
                "If ambiguous is true, ASK the user which one — never guess, "
                "because a fact filed under the wrong person looks normal forever. "
                "`similar_names` holds near-misses (typos, partial names) that did "
                "NOT match literally; treat those as suggestions, not matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional hints from the message — location, activity, "
                            "hobby, who else was there. Used to ORDER matches when "
                            "several people share a name. Never resolves ambiguity "
                            "on its own."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_person_facts",
            "description": (
                "All facts for one person. include_past=true also returns "
                "superseded facts with their end dates ('used to work at Maybank')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer"},
                    "include_past": {"type": "boolean"},
                    "category": {"type": "string", "enum": list(FACT_CATEGORIES)},
                },
                "required": ["person_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fact_provenance",
            "description": (
                "Return a fact together with the original note it came from. Use "
                "this for when / where / why questions — those details live in "
                "the raw note, not in the fact."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fact_id": {"type": "integer"}},
                "required": ["fact_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_facts",
            "description": "Search facts across ALL people, e.g. 'who likes matcha?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "enum": list(FACT_CATEGORIES)},
                    "include_past": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Search the raw notes. Use for when / where / who-was-there "
                "questions and anything the facts don't capture."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "person_id": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_relations",
            "description": "Relationships involving one person.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer"},
                    "include_past": {"type": "boolean"},
                },
                "required": ["person_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_about_me",
            "description": (
                "Save DURABLE facts about the USER themselves — things they say "
                "with 'I', 'me' or 'my'. 'I'm allergic to prawns', 'I work at "
                "Maybank', 'my sister is Mei'. Same durability bar as anyone "
                "else: 'I'm tired today' is not a fact. Creates the user's own "
                "profile on first use."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "display_name": {
                        "type": "string",
                        "description": "The user's name, if they have just told you it.",
                    },
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string", "enum": list(FACT_CATEGORIES)},
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                                "reason": {"type": "string"},
                                "source_note_id": {"type": "integer"},
                                "date_value": {"type": "string", "description": "YYYY-MM-DD"},
                                "recurring": {"type": "boolean"},
                            },
                            "required": ["category", "key", "value"],
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_about_me",
            "description": (
                "What you know about the USER: their facts and their relations. "
                "Use this when they ask what you know about them, and BEFORE "
                "giving advice that depends on their situation — what to cook "
                "for someone, what to give as a gift, whether they can eat "
                "something."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_person_brief",
            "description": (
                "Everything worth knowing before seeing someone: current facts by "
                "category, what changed recently, relations, how long since you last "
                "saw them, and their recent notes VERBATIM. Use this for 'I'm seeing "
                "X tomorrow' / 'catch me up on X'. Read the recent notes for loose "
                "ends the user never followed up on and raise them as questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer"},
                    "recent_days": {
                        "type": "integer",
                        "description": "Window for 'changed recently'. Default 90.",
                    },
                },
                "required": ["person_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reminders",
            "description": (
                "What is coming up: dated facts (birthdays, anniversaries), plans "
                "the user already wrote down as future-dated notes, and who has "
                "gone quiet. Use for 'anything coming up?', 'who should I "
                "catch up with?', or when the user asks what they are forgetting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "within_days": {"type": "integer", "description": "Default 30."},
                    "quiet_days": {
                        "type": "integer",
                        "description": "Treat someone as out of touch after this many days. Default 90.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_people",
            "description": "Every person known for this user.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

WRITE_TOOLS = {
    "save_note",
    "upsert_person",
    "save_facts",
    "save_relation",
    "correct_fact",
    "merge_people",
}


# ---------------------------------------------------------------------------
# implementations — owner_id is always the first argument, from the harness
# ---------------------------------------------------------------------------


def _save_note(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        note = repo.save_note(
            session,
            owner_id,
            raw_text=args["raw_text"],
            person_ids=args.get("person_ids") or [],
            # Injected by the harness like owner_id, never taken from the
            # model: provenance of the recording is a fact about how the
            # note arrived, not something the model gets to assert.
            source=args.get("_source", "text"),
            audio_path=args.get("_audio_path"),
            event_date=args.get("event_date"),
            location=args.get("location"),
            activity=args.get("activity"),
        )
        if note.person_ids:
            repo.touch_people(session, owner_id, note.person_ids)
        return {"ok": True, **repo.note_to_dict(note)}


def _upsert_person(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        person = repo.upsert_person(
            session,
            owner_id,
            display_name=args["display_name"],
            aliases=args.get("aliases"),
            relationship=args.get("relationship"),
            how_we_met=args.get("how_we_met"),
            disambiguator=args.get("disambiguator"),
            person_id=args.get("person_id"),
        )
        return {"ok": True, **repo.person_to_dict(person)}


def _save_facts(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    facts = args.get("facts") or []
    if isinstance(facts, dict):  # tolerate a single object instead of a list
        facts = [facts]
    results: list[dict[str, Any]] = []
    with get_session() as session:
        for item in facts:
            results.append(
                repo.save_fact(
                    session,
                    owner_id,
                    person_id=item["person_id"],
                    category=item.get("category", "preference"),
                    key=item["key"],
                    value=item["value"],
                    reason=item.get("reason"),
                    confidence=float(item.get("confidence", 1.0)),
                    source_note_id=item.get("source_note_id"),
                    valid_from=item.get("valid_from"),
                    date_value=item.get("date_value"),
                    recurring=bool(item.get("recurring", False)),
                )
            )
            # A note saved before the person was identified still needs
            # the link, and the person needs the mention credit — but only
            # once per note, however many facts came out of it.
            note_id = item.get("source_note_id")
            if note_id:
                note = repo.get_note(session, owner_id, note_id)
                if note is not None and item["person_id"] not in (note.person_ids or []):
                    repo.link_note_to_people(
                        session, owner_id, note_id, [item["person_id"]]
                    )
                    repo.touch_people(session, owner_id, [item["person_id"]])
    return {"ok": True, "saved": results}


def _save_relation(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        return repo.save_relation(
            session,
            owner_id,
            from_person=args["from_person"],
            to_person=args["to_person"],
            type=args["type"],
            valid_from=args.get("valid_from"),
            valid_to=args.get("valid_to"),
        )


def _correct_fact(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        return repo.correct_fact(
            session,
            owner_id,
            fact_id=args["fact_id"],
            new_value=args.get("new_value"),
            new_reason=args.get("new_reason"),
            move_to_person_id=args.get("move_to_person_id"),
            delete=bool(args.get("delete", False)),
        )


def _merge_people(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        return repo.merge_people(
            session, owner_id, keep_id=args["keep_id"], merge_id=args["merge_id"]
        )


def _find_person(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        matches, suggestions = repo.find_people(
            session, owner_id, args["name"], context=args.get("context", "")
        )
        ambiguous = len(matches) > 1
        return {
            "ok": True,
            "query": args["name"],
            "match_count": len(matches),
            "matches": [repo.person_to_dict(p) for p in matches],
            # Near-misses are kept OUT of matches so fuzzy recall cannot
            # manufacture ambiguity on every ordinary lookup.
            "similar_names": [
                {**repo.person_to_dict(p), "similarity": score} for p, score in suggestions
            ],
            # The model must not pick for the user when this is true.
            "ambiguous": ambiguous,
            "hint": (
                "More than one match. ASK the user which person they mean, quoting "
                "each disambiguator. Do NOT guess."
                if ambiguous
                else (
                    "No match — this person is new; create them with upsert_person."
                    if not matches
                    else "Exactly one match."
                )
            ),
        }


def _get_person_facts(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        facts = repo.get_person_facts(
            session,
            owner_id,
            person_id=args["person_id"],
            include_past=bool(args.get("include_past", False)),
            category=args.get("category"),
        )
        person = repo.get_person(session, owner_id, args["person_id"])
        return {
            "ok": True,
            "person": repo.person_to_dict(person) if person else None,
            "facts": [repo.fact_to_dict(f) for f in facts],
        }


def _get_fact_provenance(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        return repo.get_fact_provenance(session, owner_id, args["fact_id"])


def _search_facts(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        rows = repo.search_facts(
            session,
            owner_id,
            query=args.get("query", ""),
            category=args.get("category"),
            include_past=bool(args.get("include_past", False)),
            limit=int(args.get("limit", 20)),
        )
        return {
            "ok": True,
            "results": [
                {**repo.fact_to_dict(f), "person_name": p.display_name}
                for f, p in rows
            ],
        }


def _search_notes(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        notes = repo.search_notes(
            session,
            owner_id,
            query=args.get("query", ""),
            person_id=args.get("person_id"),
            limit=int(args.get("limit", 10)),
        )
        return {"ok": True, "notes": [repo.note_to_dict(n) for n in notes]}


def _get_relations(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        relations = repo.get_relations(
            session,
            owner_id,
            person_id=args["person_id"],
            include_past=bool(args.get("include_past", False)),
        )
        ids = {r.from_person for r in relations} | {r.to_person for r in relations}
        ids.add(args["person_id"])
        names = repo.names_for(session, owner_id, sorted(ids))
        return {
            "ok": True,
            # Already phrased from this person's side — use `relation` and
            # `phrase`, never re-derive direction from from/to.
            "relations": [
                repo.relation_from_perspective(r, args["person_id"], names)
                for r in relations
            ],
        }


def _remember_about_me(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    facts = args.get("facts") or []
    if isinstance(facts, dict):
        facts = [facts]
    with get_session() as session:
        me = repo.ensure_self(session, owner_id, args.get("display_name"))
        results = [
            repo.save_fact(
                session,
                owner_id,
                person_id=me.id,
                category=item.get("category", "preference"),
                key=item["key"],
                value=item["value"],
                reason=item.get("reason"),
                source_note_id=item.get("source_note_id"),
                date_value=item.get("date_value"),
                recurring=bool(item.get("recurring", False)),
            )
            for item in facts
        ]
        return {"ok": True, "me": repo.person_to_dict(me), "saved": results}


def _get_about_me(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        me = repo.get_self(session, owner_id)
        if me is None:
            return {
                "ok": True,
                "known": False,
                "hint": "Nothing about the user yet. Do not invent any.",
            }
        facts = repo.get_person_facts(session, owner_id, me.id)
        relations = repo.get_relations(session, owner_id, me.id)
        names = repo.names_for(
            session,
            owner_id,
            [r.from_person for r in relations] + [r.to_person for r in relations] + [me.id],
        )
        return {
            "ok": True,
            "known": True,
            "me": repo.person_to_dict(me),
            "facts": [repo.fact_to_dict(f) for f in facts],
            "relations": [
                repo.relation_from_perspective(r, me.id, names) for r in relations
            ],
        }


def _get_person_brief(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        return repo.get_person_brief(
            session,
            owner_id,
            person_id=args["person_id"],
            recent_days=int(args.get("recent_days", 90)),
        )


def _get_reminders(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        return {
            "ok": True,
            "upcoming": repo.get_upcoming(
                session, owner_id, within_days=int(args.get("within_days", 30))
            ),
            # Plans the user wrote down, which no fact can express.
            "planned": repo.get_planned_notes(
                session, owner_id, within_days=int(args.get("within_days", 30))
            ),
            "out_of_touch": repo.get_neglected(
                session, owner_id, quiet_days=int(args.get("quiet_days", 90))
            ),
        }


def _list_people(owner_id: str, args: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        people = repo.list_people(session, owner_id, include_self=False)
        return {"ok": True, "people": [repo.person_to_dict(p) for p in people]}


TOOL_IMPLS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "save_note": _save_note,
    "upsert_person": _upsert_person,
    "save_facts": _save_facts,
    "save_relation": _save_relation,
    "correct_fact": _correct_fact,
    "merge_people": _merge_people,
    "find_person": _find_person,
    "get_person_facts": _get_person_facts,
    "get_fact_provenance": _get_fact_provenance,
    "search_facts": _search_facts,
    "search_notes": _search_notes,
    "get_relations": _get_relations,
    "remember_about_me": _remember_about_me,
    "get_about_me": _get_about_me,
    "get_person_brief": _get_person_brief,
    "get_reminders": _get_reminders,
    "list_people": _list_people,
}


# Keys the harness injects into tool arguments. Underscore-prefixed and
# stripped from anything the model sends, so a model that guesses the name
# still cannot set them.
INJECTED_KEYS = ("_source", "_audio_path")


def dispatch(
    name: str,
    args: dict[str, Any],
    owner_id: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one tool call.

    owner_id comes from the harness and is passed positionally, so no
    value the model emits can reach it — `args` is never consulted for it.
    `context` carries the same kind of harness-owned truth (how the message
    arrived: text or voice, and where the audio lives).

    Errors are returned, not raised: the loop feeds them back to the model
    so it can recover instead of the session dying.
    """
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    # Defensive: a model that invents these must not influence the tenant
    # filter or forge the provenance of a recording.
    args = {
        k: v for k, v in args.items() if k != "owner_id" and k not in INJECTED_KEYS
    }
    if context and name == "save_note":
        args = {**args, **{k: v for k, v in context.items() if k in INJECTED_KEYS}}
    try:
        return impl(owner_id, args)
    except KeyError as exc:
        return {"ok": False, "error": f"missing required argument: {exc}"}
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
