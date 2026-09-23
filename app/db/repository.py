"""All SQL lives here. Tools never build queries.

Every function takes an explicit `owner_id` and filters on it. Facts and
relations carry no owner column of their own, so they are scoped by
joining through `people` — the tenant boundary must hold even for tables
that don't name it.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from rapidfuzz import fuzz, process

from app.db.models import KEY_ALIASES, Fact, Note, Person, Relation

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _like(term: str) -> str:
    """Escape LIKE wildcards so a literal % in a note isn't a match-all."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _parse_date(value: str | date | None) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


# Measured separation: spelling drift for the same concept scores 93-95
# (employer/employers, birth_day/birthday), while genuinely different keys
# top out at 83 (matcha/mocha, sister/mister). 90 sits in the gap with
# margin on both sides. Merging "matcha" into "mocha" would be worse than
# the bug this prevents, so the threshold errs high.
KEY_MATCH_THRESHOLD = 90
_MIN_FUZZY_LEN = 4


def normalise_key(key: str) -> str:
    """Canonical form of a fact key.

    The model invents these strings, and supersession matches on them, so
    unstable keys mean silently duplicated "current" facts. Normalisation
    is the deterministic half of the defence; resolve_key adds the rest.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_")
    return KEY_ALIASES.get(slug, slug)


def resolve_key(
    session: Session, person_id: int, category: str, key: str
) -> tuple[str, Optional[str]]:
    """Map a freshly-invented key onto an existing one where they mean the same.

    Returns (key_to_use, collapsed_from). Only keys already used for THIS
    person in THIS category are candidates, which keeps the blast radius
    of a wrong match to one person's one category.
    """
    wanted = normalise_key(key)
    if not wanted:
        return wanted, None
    # Report any rewrite of the model's own wording, not just fuzzy hits —
    # an alias collapse ("workplace" -> "employer") changes which fact gets
    # superseded, so the model needs to see that it happened.
    raw = re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_")
    rewritten = raw if raw != wanted else None

    existing = {
        normalise_key(row)
        for row in session.scalars(
            select(Fact.key).where(
                Fact.person_id == person_id,
                Fact.category == category,
                Fact.valid_to.is_(None),
            )
        ).all()
    }
    if not existing:
        return wanted, None
    if wanted in existing:
        return wanted, rewritten

    if len(wanted) >= _MIN_FUZZY_LEN:
        candidates = [k for k in existing if len(k) >= _MIN_FUZZY_LEN]
        if candidates:
            match = process.extractOne(
                wanted, candidates, scorer=fuzz.ratio, score_cutoff=KEY_MATCH_THRESHOLD
            )
            if match:
                return match[0], raw
    return wanted, rewritten


def sibling_keys(session: Session, person_id: int, category: str) -> list[str]:
    """Keys already in use for this person+category.

    Returned to the model after a write so it can notice when it has just
    created a near-duplicate slot and fix it with correct_fact.
    """
    return sorted(
        {
            row
            for row in session.scalars(
                select(Fact.key).where(
                    Fact.person_id == person_id,
                    Fact.category == category,
                    Fact.valid_to.is_(None),
                )
            ).all()
        }
    )


def person_to_dict(person: Person) -> dict[str, Any]:
    return {
        "person_id": person.id,
        "display_name": person.display_name,
        "aliases": list(person.aliases or []),
        "relationship": person.relationship,
        "how_we_met": person.how_we_met,
        "disambiguator": person.disambiguator,
        "mention_count": person.mention_count,
        "last_mentioned": person.last_mentioned.isoformat()
        if person.last_mentioned
        else None,
    }


def fact_to_dict(fact: Fact) -> dict[str, Any]:
    return {
        "fact_id": fact.id,
        "person_id": fact.person_id,
        "category": fact.category,
        "key": fact.key,
        "value": fact.value,
        "reason": fact.reason,
        "confidence": fact.confidence,
        "source_note_id": fact.source_note_id,
        "date_value": fact.date_value.isoformat() if fact.date_value else None,
        "recurring": bool(fact.recurring),
        "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
        "is_current": fact.valid_to is None,
    }


def note_to_dict(note: Note) -> dict[str, Any]:
    return {
        "note_id": note.id,
        "raw_text": note.raw_text,
        "source": note.source,
        # Surfaced so a mangled transcript can be traced back to the
        # recording. Stored but unreachable would be pointless.
        "audio_path": note.audio_path,
        "person_ids": list(note.person_ids or []),
        "event_date": note.event_date.isoformat() if note.event_date else None,
        "location": note.location,
        "activity": note.activity,
        "created_at": note.created_at.isoformat() if note.created_at else None,
    }


def relation_to_dict(relation: Relation, names: dict[int, str]) -> dict[str, Any]:
    return {
        "relation_id": relation.id,
        "from_person": relation.from_person,
        "from_name": names.get(relation.from_person),
        "to_person": relation.to_person,
        "to_name": names.get(relation.to_person),
        "type": relation.type,
        "valid_from": relation.valid_from.isoformat() if relation.valid_from else None,
        "valid_to": relation.valid_to.isoformat() if relation.valid_to else None,
        "is_current": relation.valid_to is None,
    }


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------


def get_person(session: Session, owner_id: str, person_id: int) -> Optional[Person]:
    """Fetch by id, still scoped by owner — never trust an id alone."""
    return session.scalar(
        select(Person).where(Person.id == person_id, Person.owner_id == owner_id)
    )


NAME_FUZZY_THRESHOLD = 82


def _context_score(
    session: Session, owner_id: str, person: Person, context: str
) -> float:
    """How well a person fits free-text context like "badminton in Bangsar".

    Ordering only — it never promotes anyone into the match list or
    resolves an ambiguity on the user's behalf.
    """
    if not context:
        return 0.0
    haystack = " ".join(
        filter(
            None,
            [
                person.disambiguator or "",
                person.how_we_met or "",
                person.relationship or "",
                *[
                    f"{f.key} {f.value} {f.reason or ''}"
                    for f in get_person_facts(session, owner_id, person.id)
                ],
                *[
                    f"{n.location or ''} {n.activity or ''}"
                    for n in search_notes(session, owner_id, "", person_id=person.id, limit=8)
                ],
            ],
        )
    ).lower()
    if not haystack.strip():
        return 0.0

    terms = [t for t in re.split(r"[^a-z0-9]+", context.lower()) if len(t) > 2]
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t in haystack)
    return 3.0 * (hits / len(terms))


def _recency_score(person: Person) -> float:
    """Recent and frequent people rank first. Ordering only."""
    score = min((person.mention_count or 0), 20) / 20.0
    if person.last_mentioned:
        days = (datetime.now(timezone.utc) - person.last_mentioned.replace(
            tzinfo=timezone.utc
        )).days
        score += max(0.0, 1.0 - days / 365.0)
    return score


def find_people(
    session: Session,
    owner_id: str,
    name: str,
    context: str = "",
) -> tuple[list[Person], list[tuple[Person, int]]]:
    """Return (strong_matches, fuzzy_suggestions).

    Strong matches are exact alias hits and substring hits on the display
    name. Fuzzy hits are returned SEPARATELY and never silently join the
    match list, because fuzzy matching raises recall — folding it in would
    invent ambiguity on every lookup and train the user to dismiss the
    "which one?" question, which is the guard rail that matters most here.
    Fuzzy results are promoted only when nothing matched strongly, which
    is exactly the typo case they exist for.

    A fact filed under the wrong Peter is the one error that is invisible
    on review, so this function never collapses to a single best guess.
    """
    needle = _norm(name)
    if not needle:
        return [], []

    candidates = session.scalars(
        select(Person).where(Person.owner_id == owner_id)
    ).all()

    exact: list[Person] = []
    substring: list[Person] = []
    fuzzy: list[tuple[Person, int]] = []

    for person in candidates:
        names = {_norm(a) for a in (person.aliases or [])}
        names.add(_norm(person.display_name))
        if needle in names:
            exact.append(person)
            continue
        if needle in _norm(person.display_name):
            substring.append(person)
            continue
        best = max(
            (fuzz.ratio(needle, n) for n in names if n),
            default=0,
        )
        # token_set catches "peter tan" vs "tan peter" and partial names.
        best = max(best, max((fuzz.token_set_ratio(needle, n) for n in names if n), default=0))
        if best >= NAME_FUZZY_THRESHOLD:
            fuzzy.append((person, int(best)))

    def rank(person: Person) -> tuple[float, int]:
        return (
            -(_context_score(session, owner_id, person, context) + _recency_score(person)),
            person.id,
        )

    exact.sort(key=rank)
    substring.sort(key=rank)
    fuzzy.sort(key=lambda pair: (-pair[1], rank(pair[0])))

    strong = exact + substring
    if not strong and fuzzy:
        # Nothing matched literally: the typo case fuzzy exists for.
        return [p for p, _ in fuzzy], []
    return strong, fuzzy


def upsert_person(
    session: Session,
    owner_id: str,
    display_name: str,
    aliases: Sequence[str] | None = None,
    relationship: str | None = None,
    how_we_met: str | None = None,
    disambiguator: str | None = None,
    person_id: int | None = None,
) -> Person:
    """Create a person, or update one identified by explicit person_id.

    Without an explicit id this only reuses a row on an exact alias match.
    Matching loosely here would merge two different Peters by accident,
    which is the one mistake that cannot be spotted later.
    """
    person: Optional[Person] = None
    if person_id is not None:
        person = get_person(session, owner_id, person_id)

    if person is None:
        strong, _ = find_people(session, owner_id, display_name)
        # Only an exact name/alias hit may reuse an existing row. Fuzzy
        # matches are deliberately ignored here: merging two different
        # Peters on a near-miss is unrecoverable.
        exact = [
            p
            for p in strong
            if _norm(display_name)
            in {_norm(a) for a in (p.aliases or [])} | {_norm(p.display_name)}
        ]
        if len(exact) == 1:
            person = exact[0]

    if person is None:
        person = Person(
            owner_id=owner_id,
            display_name=display_name.strip(),
            aliases=sorted({_norm(a) for a in (aliases or []) if _norm(a)}),
            relationship=relationship,
            how_we_met=how_we_met,
            disambiguator=disambiguator,
            mention_count=0,
        )
        session.add(person)
        session.flush()
        return person

    # Update: fill blanks and add aliases, never clear an existing value
    # with a None the model happened to omit this turn.
    if display_name and display_name.strip():
        person.display_name = display_name.strip()
    if aliases:
        merged = {_norm(a) for a in (person.aliases or [])}
        merged |= {_norm(a) for a in aliases if _norm(a)}
        person.aliases = sorted(merged)
    if relationship:
        person.relationship = relationship
    if how_we_met:
        person.how_we_met = how_we_met
    if disambiguator:
        person.disambiguator = disambiguator
    session.flush()
    return person


def list_people(session: Session, owner_id: str) -> list[Person]:
    return list(
        session.scalars(
            select(Person)
            .where(Person.owner_id == owner_id)
            .order_by(Person.mention_count.desc(), Person.display_name)
        ).all()
    )


def touch_people(
    session: Session, owner_id: str, person_ids: Sequence[int]
) -> None:
    """Bump mention_count / last_mentioned — feeds phase-4 context scoring."""
    for pid in person_ids:
        person = get_person(session, owner_id, pid)
        if person is not None:
            person.mention_count = (person.mention_count or 0) + 1
            person.last_mentioned = datetime.now(timezone.utc)
    session.flush()


def merge_people(
    session: Session, owner_id: str, keep_id: int, merge_id: int
) -> dict[str, Any]:
    """Fold `merge_id` into `keep_id`, moving every row that referenced it.

    Data is relocated, never dropped — the duplicate row disappears only
    once nothing points at it.
    """
    keep = get_person(session, owner_id, keep_id)
    merge = get_person(session, owner_id, merge_id)
    if keep is None or merge is None:
        return {"ok": False, "error": "person not found"}
    if keep.id == merge.id:
        return {"ok": False, "error": "cannot merge a person into themselves"}

    moved_facts = 0
    for fact in session.scalars(select(Fact).where(Fact.person_id == merge.id)).all():
        fact.person_id = keep.id
        moved_facts += 1

    for relation in session.scalars(
        select(Relation).where(
            or_(Relation.from_person == merge.id, Relation.to_person == merge.id)
        )
    ).all():
        if relation.from_person == merge.id:
            relation.from_person = keep.id
        if relation.to_person == merge.id:
            relation.to_person = keep.id

    moved_notes = 0
    for note in session.scalars(
        select(Note).where(Note.owner_id == owner_id)
    ).all():
        ids = list(note.person_ids or [])
        if merge.id in ids:
            # Reassign the reference; the note's raw_text is untouched.
            note.person_ids = sorted({keep.id if i == merge.id else i for i in ids})
            moved_notes += 1

    aliases = {_norm(a) for a in (keep.aliases or [])}
    aliases |= {_norm(a) for a in (merge.aliases or [])}
    aliases.add(_norm(merge.display_name))
    keep.aliases = sorted(a for a in aliases if a)
    keep.mention_count = (keep.mention_count or 0) + (merge.mention_count or 0)
    keep.how_we_met = keep.how_we_met or merge.how_we_met
    keep.relationship = keep.relationship or merge.relationship
    keep.disambiguator = keep.disambiguator or merge.disambiguator

    session.delete(merge)
    session.flush()
    return {
        "ok": True,
        "kept": person_to_dict(keep),
        "moved_facts": moved_facts,
        "moved_notes": moved_notes,
    }


# ---------------------------------------------------------------------------
# notes  (layer 1 — append only)
# ---------------------------------------------------------------------------


def save_note(
    session: Session,
    owner_id: str,
    raw_text: str,
    person_ids: Sequence[int] | None = None,
    source: str = "text",
    audio_path: str | None = None,
    event_date: str | date | None = None,
    location: str | None = None,
    activity: str | None = None,
) -> Note:
    """Insert a note verbatim. There is deliberately no update_note."""
    note = Note(
        owner_id=owner_id,
        raw_text=raw_text,
        source=source,
        audio_path=audio_path,
        person_ids=sorted(set(person_ids or [])),
        event_date=_parse_date(event_date),
        location=location,
        activity=activity,
    )
    session.add(note)
    session.flush()
    return note


def get_note(session: Session, owner_id: str, note_id: int) -> Optional[Note]:
    return session.scalar(
        select(Note).where(Note.id == note_id, Note.owner_id == owner_id)
    )


def link_note_to_people(
    session: Session, owner_id: str, note_id: int, person_ids: Sequence[int]
) -> Optional[Note]:
    """Attach people to a note saved before they were identified.

    This edits person_ids only — an index, not content. raw_text stays
    immutable, which is what makes facts safely re-derivable.
    """
    note = get_note(session, owner_id, note_id)
    if note is None:
        return None
    note.person_ids = sorted(set(note.person_ids or []) | set(person_ids))
    session.flush()
    return note


def _fts_available(session: Session) -> bool:
    """Is the FTS5 index usable on this connection?

    FTS5 is a SQLite compile-time option and simply absent on Postgres, so
    every caller must be able to fall back. Cached per engine because this
    runs on every search.
    """
    bind = session.get_bind()
    cached = getattr(bind, "_relmem_fts", None)
    if cached is not None:
        return cached
    ok = False
    if bind.dialect.name == "sqlite":
        try:
            session.execute(text("SELECT 1 FROM notes_fts LIMIT 1"))
            ok = True
        except Exception:
            ok = False
    try:
        bind._relmem_fts = ok
    except AttributeError:
        pass
    return ok


def search_notes(
    session: Session,
    owner_id: str,
    query: str = "",
    person_id: int | None = None,
    limit: int = 10,
) -> list[Note]:
    """Full-text search over raw notes, newest first.

    Uses SQLite FTS5 where available: it stems, ranks by relevance and
    handles multi-word queries, none of which LIKE can do — "matcha Kyoto"
    should find the note whether or not those words sit next to each
    other. Falls back to LIKE on Postgres or a SQLite build without FTS5,
    behind this unchanged signature.

    TODO: on Postgres, swap the fallback for tsvector, and add pgvector
    embeddings for genuinely semantic recall ("who did I talk to about
    career stuff") — still behind this signature.
    """
    fetch = limit * 5 if person_id else limit
    notes: list[Note] = []

    if query and _fts_available(session):
        try:
            rows = session.execute(
                text(
                    "SELECT rowid FROM notes_fts WHERE notes_fts MATCH :q "
                    "ORDER BY rank LIMIT :lim"
                ),
                {"q": _fts_query(query), "lim": fetch},
            ).all()
            ids = [r[0] for r in rows]
            if ids:
                found = session.scalars(
                    select(Note).where(Note.id.in_(ids), Note.owner_id == owner_id)
                ).all()
                order = {note_id: i for i, note_id in enumerate(ids)}
                notes = sorted(found, key=lambda n: order.get(n.id, 1_000_000))
        except Exception:
            notes = []  # malformed FTS expression: fall through to LIKE

    if not notes:
        stmt = select(Note).where(Note.owner_id == owner_id)
        if query:
            stmt = stmt.where(Note.raw_text.like(_like(query), escape="\\"))
        stmt = stmt.order_by(Note.event_date.desc().nulls_last(), Note.id.desc())
        notes = list(session.scalars(stmt.limit(fetch)).all())

    if person_id is not None:
        # JSON list membership isn't portable across SQLite/Postgres, so
        # filter in Python rather than write dialect-specific SQL here.
        notes = [n for n in notes if person_id in (n.person_ids or [])]
    return notes[:limit]


def _fts_query(query: str) -> str:
    """Turn user text into a safe FTS5 expression.

    Raw input goes straight into MATCH, where a stray quote or a bare
    "AND" is a syntax error, so every token is quoted and ORed.
    """
    tokens = [t for t in re.split(r"[^\w]+", query, flags=re.UNICODE) if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


# ---------------------------------------------------------------------------
# facts  (layer 2 — superseded, never overwritten)
# ---------------------------------------------------------------------------


def get_current_fact(
    session: Session, person_id: int, key: str
) -> Optional[Fact]:
    return session.scalar(
        select(Fact)
        .where(Fact.person_id == person_id, Fact.key == key, Fact.valid_to.is_(None))
        .order_by(Fact.id.desc())
    )


def save_fact(
    session: Session,
    owner_id: str,
    person_id: int,
    category: str,
    key: str,
    value: str,
    reason: str | None = None,
    confidence: float = 1.0,
    source_note_id: int | None = None,
    valid_from: str | date | None = None,
    date_value: str | date | None = None,
    recurring: bool = False,
) -> dict[str, Any]:
    """Write one fact, superseding any current fact with the same key.

    The old row is closed (valid_to + superseded_by), never deleted, so
    "he used to work at Maybank" survives the move to Grab.

    The key is normalised and resolved against keys already in use first.
    Without that, the model writing "employer" once and "workplace" later
    leaves BOTH facts current, and the app reports two jobs as though the
    person held them at the same time — a silent, normal-looking wrong
    answer, which is the failure mode this whole design exists to avoid.
    """
    person = get_person(session, owner_id, person_id)
    if person is None:
        return {"ok": False, "error": f"no person {person_id} for this owner"}

    key, collapsed_from = resolve_key(session, person_id, category, key)

    # Dates follow the note's event_date, not the wall clock: writing up
    # Tuesday's lunch on Thursday must still date the fact to Tuesday.
    effective = _parse_date(valid_from)
    if effective is None and source_note_id is not None:
        note = get_note(session, owner_id, source_note_id)
        if note is not None:
            effective = note.event_date

    existing = get_current_fact(session, person_id, key)
    if existing is not None and _norm(existing.value) == _norm(value):
        if date_value and not existing.date_value:
            existing.date_value = _parse_date(date_value)
            existing.recurring = bool(recurring)
        # Re-stating the same thing is not new information. Writing a
        # duplicate row would make the history look like a change.
        if reason and not existing.reason:
            existing.reason = reason
        session.flush()
        return {"ok": True, "action": "unchanged", "fact": fact_to_dict(existing)}

    fact = Fact(
        person_id=person_id,
        category=category,
        key=key,
        value=value,
        reason=reason,
        confidence=confidence,
        source_note_id=source_note_id,
        valid_from=effective,
        date_value=_parse_date(date_value),
        recurring=bool(recurring),
    )
    session.add(fact)
    session.flush()

    action = "created"
    superseded: Optional[dict[str, Any]] = None
    if existing is not None:
        existing.valid_to = effective or date.today()
        existing.superseded_by = fact.id
        action = "superseded"
        superseded = fact_to_dict(existing)

    session.flush()
    result: dict[str, Any] = {
        "ok": True,
        "action": action,
        "fact": fact_to_dict(fact),
        "superseded": superseded,
    }
    if collapsed_from and collapsed_from != key:
        result["note_to_model"] = (
            f"Key {collapsed_from!r} was treated as the existing key {key!r}"
            + (
                " so the older value was superseded rather than left running "
                "alongside it."
                if action == "superseded"
                else ", the canonical name for this slot. Use that key next time."
            )
        )
    if "note_to_model" not in result and action == "created":
        others = [k for k in sibling_keys(session, person_id, category) if k != key]
        if others:
            # Surfaced so the model can spot a near-duplicate slot it just
            # created and fix it with correct_fact, rather than leaving two
            # facts that both claim to be current.
            result["sibling_keys"] = others
            result["note_to_model"] = (
                f"New key {key!r} created in category {category!r}, which already "
                f"uses: {', '.join(others)}. If this is the same thing under a "
                "different name, fix it with correct_fact."
            )
    return result


def get_person_facts(
    session: Session,
    owner_id: str,
    person_id: int,
    include_past: bool = False,
    category: str | None = None,
) -> list[Fact]:
    person = get_person(session, owner_id, person_id)
    if person is None:
        return []
    stmt = select(Fact).where(Fact.person_id == person_id)
    if not include_past:
        stmt = stmt.where(Fact.valid_to.is_(None))
    if category:
        stmt = stmt.where(Fact.category == category)
    return list(
        session.scalars(stmt.order_by(Fact.category, Fact.key, Fact.id)).all()
    )


def get_fact(session: Session, owner_id: str, fact_id: int) -> Optional[Fact]:
    """Fetch a fact, scoped by joining through people to the owner."""
    return session.scalar(
        select(Fact)
        .join(Person, Person.id == Fact.person_id)
        .where(Fact.id == fact_id, Person.owner_id == owner_id)
    )


def get_fact_provenance(
    session: Session, owner_id: str, fact_id: int
) -> dict[str, Any]:
    """Return the fact together with the raw note it came from.

    This is the whole reason source_note_id exists: the fact answers
    *what*, only the note answers *when, where and why*.
    """
    fact = get_fact(session, owner_id, fact_id)
    if fact is None:
        return {"ok": False, "error": f"no fact {fact_id}"}

    person = get_person(session, owner_id, fact.person_id)
    payload: dict[str, Any] = {
        "ok": True,
        "fact": fact_to_dict(fact),
        "person_name": person.display_name if person else None,
        "note": None,
    }
    if fact.source_note_id is not None:
        note = get_note(session, owner_id, fact.source_note_id)
        if note is not None:
            payload["note"] = note_to_dict(note)
    return payload


def search_facts(
    session: Session,
    owner_id: str,
    query: str = "",
    category: str | None = None,
    include_past: bool = False,
    limit: int = 20,
) -> list[tuple[Fact, Person]]:
    """Cross-person fact search. Joined to people for the owner filter."""
    stmt = (
        select(Fact, Person)
        .join(Person, Person.id == Fact.person_id)
        .where(Person.owner_id == owner_id)
    )
    if query:
        pattern = _like(query)
        stmt = stmt.where(
            or_(
                Fact.value.like(pattern, escape="\\"),
                Fact.key.like(pattern, escape="\\"),
                Fact.reason.like(pattern, escape="\\"),
            )
        )
    if category:
        stmt = stmt.where(Fact.category == category)
    if not include_past:
        stmt = stmt.where(Fact.valid_to.is_(None))
    rows = session.execute(stmt.order_by(Fact.id.desc()).limit(limit)).all()
    return [(row[0], row[1]) for row in rows]


def correct_fact(
    session: Session,
    owner_id: str,
    fact_id: int,
    new_value: str | None = None,
    new_reason: str | None = None,
    move_to_person_id: int | None = None,
    delete: bool = False,
) -> dict[str, Any]:
    """Fix a bad extraction.

    Distinct from supersession: supersession records that the world
    changed, correction records that the extraction was wrong. A wrong
    row has no history worth keeping, so here editing and deleting are
    allowed — the note it came from is still intact either way.
    """
    fact = get_fact(session, owner_id, fact_id)
    if fact is None:
        return {"ok": False, "error": f"no fact {fact_id}"}

    if delete:
        session.delete(fact)
        session.flush()
        return {"ok": True, "action": "deleted", "fact_id": fact_id}

    actions: list[str] = []
    if move_to_person_id is not None:
        target = get_person(session, owner_id, move_to_person_id)
        if target is None:
            return {"ok": False, "error": f"no person {move_to_person_id}"}
        fact.person_id = target.id
        actions.append(f"moved to {target.display_name}")
    if new_value is not None:
        fact.value = new_value
        actions.append("value updated")
    if new_reason is not None:
        fact.reason = new_reason
        actions.append("reason updated")

    session.flush()
    return {
        "ok": True,
        "action": ", ".join(actions) or "no change",
        "fact": fact_to_dict(fact),
    }


def delete_facts_for_owner(session: Session, owner_id: str) -> int:
    """Wipe layer 2 so it can be re-derived from the immutable notes.

    Exposed for maintenance, never as a tool — the model has no business
    clearing memory.
    """
    facts = session.scalars(
        select(Fact).join(Person, Person.id == Fact.person_id).where(
            Person.owner_id == owner_id
        )
    ).all()
    for fact in facts:
        session.delete(fact)
    session.flush()
    return len(facts)


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------


def save_relation(
    session: Session,
    owner_id: str,
    from_person: int,
    to_person: int,
    type: str,
    valid_from: str | date | None = None,
    valid_to: str | date | None = None,
) -> dict[str, Any]:
    a = get_person(session, owner_id, from_person)
    b = get_person(session, owner_id, to_person)
    if a is None or b is None:
        return {"ok": False, "error": "both people must exist for this owner"}

    existing = session.scalar(
        select(Relation).where(
            Relation.from_person == from_person,
            Relation.to_person == to_person,
            Relation.type == type,
            Relation.valid_to.is_(None),
        )
    )
    if existing is not None:
        if valid_to is not None:
            existing.valid_to = _parse_date(valid_to)
            session.flush()
            return {"ok": True, "action": "ended", "relation_id": existing.id}
        return {"ok": True, "action": "unchanged", "relation_id": existing.id}

    relation = Relation(
        from_person=from_person,
        to_person=to_person,
        type=type,
        valid_from=_parse_date(valid_from),
        valid_to=_parse_date(valid_to),
    )
    session.add(relation)
    session.flush()
    return {"ok": True, "action": "created", "relation_id": relation.id}


def get_relations(
    session: Session, owner_id: str, person_id: int, include_past: bool = False
) -> list[Relation]:
    person = get_person(session, owner_id, person_id)
    if person is None:
        return []
    stmt = select(Relation).where(
        or_(Relation.from_person == person_id, Relation.to_person == person_id)
    )
    if not include_past:
        stmt = stmt.where(Relation.valid_to.is_(None))
    return list(session.scalars(stmt.order_by(Relation.id)).all())


def names_for(session: Session, owner_id: str, ids: Sequence[int]) -> dict[int, str]:
    if not ids:
        return {}
    rows = session.scalars(
        select(Person).where(Person.owner_id == owner_id, Person.id.in_(list(ids)))
    ).all()
    return {p.id: p.display_name for p in rows}


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def stats(session: Session, owner_id: str) -> dict[str, int]:
    people = session.scalar(
        select(func.count(Person.id)).where(Person.owner_id == owner_id)
    )
    notes = session.scalar(
        select(func.count(Note.id)).where(Note.owner_id == owner_id)
    )
    current = session.scalar(
        select(func.count(Fact.id))
        .join(Person, Person.id == Fact.person_id)
        .where(Person.owner_id == owner_id, Fact.valid_to.is_(None))
    )
    past = session.scalar(
        select(func.count(Fact.id))
        .join(Person, Person.id == Fact.person_id)
        .where(Person.owner_id == owner_id, Fact.valid_to.is_not(None))
    )
    relations = session.scalar(
        select(func.count(Relation.id))
        .join(Person, Person.id == Relation.from_person)
        .where(Person.owner_id == owner_id)
    )
    return {
        "people": people or 0,
        "notes": notes or 0,
        "current_facts": current or 0,
        "past_facts": past or 0,
        "relations": relations or 0,
    }


# ---------------------------------------------------------------------------
# brief  (what to know before seeing someone)
# ---------------------------------------------------------------------------


def get_person_brief(
    session: Session, owner_id: str, person_id: int, recent_days: int = 90
) -> dict[str, Any]:
    """Everything worth knowing before meeting one person.

    Deliberately returns raw recent notes alongside the facts. Spotting an
    open thread ("in June he was interviewing — did that happen?") is a
    judgement call, so the database does retrieval and the model does the
    judging. Encoding it as a keyword heuristic here would be brittle and
    would hide its own mistakes.
    """
    person = get_person(session, owner_id, person_id)
    if person is None:
        return {"ok": False, "error": f"no person {person_id}"}

    facts = get_person_facts(session, owner_id, person_id, include_past=True)
    current = [f for f in facts if f.valid_to is None]
    cutoff = date.today() - timedelta(days=recent_days)

    # "Changed recently" means either newly true or newly untrue, so that a
    # job he just left is as visible as the one he just started.
    changed = [
        f
        for f in facts
        if (f.valid_from and f.valid_from >= cutoff)
        or (f.valid_to and f.valid_to >= cutoff)
    ]

    notes = search_notes(session, owner_id, "", person_id=person_id, limit=6)
    last_seen = next((n.event_date for n in notes if n.event_date), None)
    relations = get_relations(session, owner_id, person_id)
    rel_names = names_for(
        session,
        owner_id,
        [r.from_person for r in relations] + [r.to_person for r in relations],
    )

    by_category: dict[str, list[dict[str, Any]]] = {}
    for f in current:
        by_category.setdefault(f.category, []).append(fact_to_dict(f))

    return {
        "ok": True,
        "person": person_to_dict(person),
        "facts_by_category": by_category,
        "changed_recently": [fact_to_dict(f) for f in changed],
        "relations": [relation_to_dict(r, rel_names) for r in relations],
        # Verbatim, so the model can quote them and spot loose ends.
        "recent_notes": [note_to_dict(n) for n in notes],
        "last_seen": last_seen.isoformat() if last_seen else None,
        "days_since_last_seen": (date.today() - last_seen).days if last_seen else None,
    }


# ---------------------------------------------------------------------------
# reminders  (the app speaking first)
# ---------------------------------------------------------------------------


def _next_occurrence(when: date, today: date) -> date:
    """Next anniversary of a date, this year or next."""
    try:
        this_year = when.replace(year=today.year)
    except ValueError:  # 29 Feb in a non-leap year
        this_year = when.replace(year=today.year, day=28)
    if this_year >= today:
        return this_year
    try:
        return when.replace(year=today.year + 1)
    except ValueError:
        return when.replace(year=today.year + 1, day=28)


def get_upcoming(
    session: Session, owner_id: str, within_days: int = 30
) -> list[dict[str, Any]]:
    """Dated facts falling due soon — birthdays, anniversaries, start dates."""
    today = date.today()
    horizon = today + timedelta(days=within_days)

    rows = session.execute(
        select(Fact, Person)
        .join(Person, Person.id == Fact.person_id)
        .where(
            Person.owner_id == owner_id,
            Fact.valid_to.is_(None),
            Fact.date_value.is_not(None),
        )
    ).all()

    upcoming: list[dict[str, Any]] = []
    for fact, person in rows:
        due = _next_occurrence(fact.date_value, today) if fact.recurring else fact.date_value
        if due < today or due > horizon:
            continue
        upcoming.append(
            {
                **fact_to_dict(fact),
                "person_name": person.display_name,
                "due": due.isoformat(),
                "days_away": (due - today).days,
            }
        )
    return sorted(upcoming, key=lambda item: item["days_away"])


def get_neglected(
    session: Session, owner_id: str, quiet_days: int = 90, limit: int = 5
) -> list[dict[str, Any]]:
    """People you haven't mentioned in a while.

    last_mentioned has been collected since day one and never read until
    now; this is what it was for. Only people mentioned more than once
    qualify, so a one-off acquaintance doesn't generate guilt forever.
    """
    today = date.today()
    people = session.scalars(
        select(Person).where(
            Person.owner_id == owner_id,
            Person.last_mentioned.is_not(None),
            Person.mention_count > 1,
        )
    ).all()

    out: list[dict[str, Any]] = []
    for person in people:
        days = (today - person.last_mentioned.date()).days
        if days >= quiet_days:
            out.append(
                {
                    **person_to_dict(person),
                    "days_quiet": days,
                    "last_mentioned_date": person.last_mentioned.date().isoformat(),
                }
            )
    return sorted(out, key=lambda item: -item["days_quiet"])[:limit]
