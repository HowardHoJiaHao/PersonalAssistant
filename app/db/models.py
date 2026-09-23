"""SQLAlchemy 2.0 declarative models — the two memory layers.

List-valued columns use JSON rather than the Postgres ARRAY type so that
these exact classes run unchanged on SQLite today and Postgres later.

Layer 1 (`Note`) is verbatim and immutable. Layer 2 (`Fact`) is distilled
and disposable: because notes are never edited, facts can be wiped and
re-derived from notes after a prompt improvement without losing anything.
Preserving that property is why no fact is ever the only copy of a detail.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# Closed vocabularies. Kept as plain tuples rather than SQL enums so that
# adding a category is a code change, not a migration on both dialects.
FACT_CATEGORIES = (
    "food",
    "hobby",
    "work",
    "family",
    "dislike",
    "milestone",
    "health",
    "preference",
)

# Supersession matches on (person_id, key), and `key` is a string the model
# invents. If it writes "employer" in March and "workplace" in June, nothing
# supersedes and BOTH show as current — silently, and looking perfectly normal.
# These aliases collapse the drift we can predict; resolve_key() in the
# repository catches the rest by fuzzy match within a category.
KEY_ALIASES: dict[str, str] = {
    "workplace": "employer",
    "work": "employer",
    "job": "employer",
    "company": "employer",
    "works_at": "employer",
    "employment": "employer",
    "occupation": "job_title",
    "role": "job_title",
    "position": "job_title",
    "title": "job_title",
    "lives_in": "location",
    "city": "location",
    "residence": "location",
    "address": "location",
    "home": "location",
    "partner": "relationship_status",
    "marital_status": "relationship_status",
    "spouse_name": "relationship_status",
    "bday": "birthday",
    "date_of_birth": "birthday",
    "dob": "birthday",
    "tel": "phone",
    "phone_number": "phone",
    "mobile": "phone",
}

RELATION_TYPES = (
    "spouse",
    "sibling",
    "parent",
    "child",
    "colleague",
    "friend",
    "introduced_by",
)


class Person(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # The tenant boundary. Indexed because every single query filters on it.
    owner_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Lowercase strings; find_person compares against these directly so the
    # normalisation happens once on write instead of on every read.
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    relationship: Mapped[Optional[str]] = mapped_column(String(50))
    how_we_met: Mapped[Optional[str]] = mapped_column(Text)
    # Short human tag ("badminton, Bangsar") shown when asking the user
    # which Peter they mean. Exists purely to make disambiguation answerable.
    disambiguator: Mapped[Optional[str]] = mapped_column(String(200))
    mention_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_mentioned: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (Index("ix_people_owner_name", "owner_id", "display_name"),)

    def __repr__(self) -> str:
        return f"<Person id={self.id} {self.display_name!r}>"


class Note(Base):
    """Layer 1: the user's exact words. Written once, never updated."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(20), default="text", nullable=False)
    # Set once voice input lands; the transcript still goes in raw_text so
    # every downstream query keeps working regardless of source.
    audio_path: Mapped[Optional[str]] = mapped_column(String(500))
    person_ids: Mapped[list[int]] = mapped_column(JSON, default=list, nullable=False)
    # When the thing happened, not when it was written down. Facts inherit
    # this, so writing up Tuesday's lunch on Thursday still dates to Tuesday.
    event_date: Mapped[Optional[date]] = mapped_column(Date)
    location: Mapped[Optional[str]] = mapped_column(String(200))
    activity: Mapped[Optional[str]] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    def __repr__(self) -> str:
        return f"<Note id={self.id} {self.raw_text[:40]!r}>"


class Fact(Base):
    """Layer 2: one distilled claim about one person.

    Rows are never overwritten. A new value for the same (person_id, key)
    closes the old row with valid_to + superseded_by, so "he used to work
    at Maybank" stays answerable after he moves to Grab.
    """

    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("people.id"), index=True, nullable=False
    )
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    # The stable slot ("employer", "matcha"). Supersession keys off this.
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    # The stated "why" ("got into it in Kyoto") when the user gave one.
    reason: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(default=1.0, nullable=False)
    # Provenance. Without this link, when/where/why questions are
    # unanswerable — only the raw note holds those details.
    source_note_id: Mapped[Optional[int]] = mapped_column(ForeignKey("notes.id"))
    # A date the fact is ABOUT (birthday, anniversary, when they moved).
    # Distinct from valid_from, which is when the claim started being true:
    # a birthday fact recorded today has valid_from=today, date_value=1992-04-03.
    date_value: Mapped[Optional[date]] = mapped_column(Date)
    # Set for dates that come round again each year, so reminders can ask
    # "whose birthday is next week?" without re-parsing free text.
    recurring: Mapped[bool] = mapped_column(default=False, nullable=False)
    valid_from: Mapped[Optional[date]] = mapped_column(Date)
    # NULL means currently true. Set when superseded.
    valid_to: Mapped[Optional[date]] = mapped_column(Date)
    superseded_by: Mapped[Optional[int]] = mapped_column(ForeignKey("facts.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (Index("ix_facts_person_key", "person_id", "key"),)

    def __repr__(self) -> str:
        return f"<Fact id={self.id} {self.key}={self.value!r}>"


class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_person: Mapped[int] = mapped_column(
        ForeignKey("people.id"), index=True, nullable=False
    )
    to_person: Mapped[int] = mapped_column(
        ForeignKey("people.id"), index=True, nullable=False
    )
    type: Mapped[str] = mapped_column(String(30), nullable=False)
    # Relationships end too; a closed relation is history, not a deletion.
    valid_from: Mapped[Optional[date]] = mapped_column(Date)
    valid_to: Mapped[Optional[date]] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    def __repr__(self) -> str:
        return f"<Relation {self.from_person}-{self.type}->{self.to_person}>"
