"""Database seam — the only file that knows which database is in use.

Moving to Postgres is a DATABASE_URL change and nothing else: no model
edits, no repository edits. That is the whole point of this file, so
keep dialect-specific handling here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db.models import Base


def _make_engine(database_url: str, echo: bool = False) -> Engine:
    kwargs: dict = {"echo": echo, "future": True}
    if database_url.startswith("sqlite"):
        # The CLI and the agent loop touch the session from asyncio's
        # thread pool, so the default same-thread check would spuriously
        # fail. Postgres needs neither of these arguments.
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            # An in-memory database lives inside its connection, so the
            # default pool would hand out a different, empty database on
            # every checkout. StaticPool keeps one shared connection.
            kwargs["poolclass"] = StaticPool
    return create_engine(database_url, **kwargs)


engine: Engine = _make_engine(settings.database_url, settings.sql_echo)
SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(target_engine: Engine | None = None) -> None:
    """Create tables if absent, then add any columns a newer build expects.

    Safe to call on every start.
    """
    target = target_engine or engine
    Base.metadata.create_all(target)
    _add_missing_columns(target)
    _setup_fts(target)


# FTS5 indexes raw_text only. It is an external-content index, so the notes
# table stays the single copy of the words and the index is pure derived
# data — it can be dropped and rebuilt without touching layer 1, exactly
# like facts can.
_FTS_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5("
    # porter stems English so "running" finds "run"; unicode61 does the
    # tokenising, which is what keeps Malay and Chinese notes searchable
    # (porter simply leaves those tokens alone).
    "  raw_text, content='notes', content_rowid='id',"
    "  tokenize='porter unicode61')",
    "CREATE TRIGGER IF NOT EXISTS notes_fts_ai AFTER INSERT ON notes BEGIN"
    "  INSERT INTO notes_fts(rowid, raw_text) VALUES (new.id, new.raw_text);"
    " END",
    "CREATE TRIGGER IF NOT EXISTS notes_fts_ad AFTER DELETE ON notes BEGIN"
    "  INSERT INTO notes_fts(notes_fts, rowid, raw_text)"
    "  VALUES ('delete', old.id, old.raw_text);"
    " END",
    "CREATE TRIGGER IF NOT EXISTS notes_fts_au AFTER UPDATE ON notes BEGIN"
    "  INSERT INTO notes_fts(notes_fts, rowid, raw_text)"
    "  VALUES ('delete', old.id, old.raw_text);"
    "  INSERT INTO notes_fts(rowid, raw_text) VALUES (new.id, new.raw_text);"
    " END",
)


def _setup_fts(target: Engine) -> bool:
    """Build the SQLite full-text index, if this build supports FTS5.

    Returns False rather than raising when FTS5 is missing or the dialect
    is Postgres — search_notes falls back to LIKE, so the app still works,
    just with worse recall. A search feature must never stop the app from
    starting.
    """
    if target.dialect.name != "sqlite":
        return False
    try:
        with target.begin() as conn:
            # The index is derived data, so an older one built with a
            # different tokenizer is dropped and rebuilt rather than
            # migrated — nothing is lost, the notes are the source.
            existing = conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='notes_fts'"
            ).scalar()
            stale_tokenizer = bool(existing) and "porter" not in existing
            if stale_tokenizer:
                conn.exec_driver_sql("DROP TABLE IF EXISTS notes_fts")
            for statement in _FTS_DDL:
                conn.exec_driver_sql(statement)

            # Backfill notes written before the index existed.
            #
            # Do NOT test this with COUNT(*) FROM notes_fts: this is an
            # external-content table, so COUNT reads the notes table and
            # returns a healthy number even when the index holds nothing.
            # That reads as "already populated" and skips the rebuild,
            # leaving every search to fall through to LIKE silently — the
            # index looks fine and simply never matches. Ask the index
            # itself instead.
            needs_backfill = (not existing) or stale_tokenizer
            if not needs_backfill:
                indexed = conn.exec_driver_sql(
                    "SELECT COUNT(*) FROM notes_fts_data"
                ).scalar()
                needs_backfill = (indexed or 0) <= 1
            if needs_backfill:
                conn.exec_driver_sql(
                    "INSERT INTO notes_fts(notes_fts) VALUES('rebuild')"
                )
        return True
    except Exception:
        return False


def rebuild_fts(target_engine: Engine | None = None) -> bool:
    """Re-derive the search index from the notes table."""
    target = target_engine or engine
    if target.dialect.name != "sqlite":
        return False
    try:
        with target.begin() as conn:
            conn.exec_driver_sql("INSERT INTO notes_fts(notes_fts) VALUES('rebuild')")
        return True
    except Exception:
        return False


def _add_missing_columns(target: Engine) -> list[str]:
    """Bring an existing database up to the current model definition.

    A real migration tool is overkill for a single-file personal database,
    but silently failing to start because an old memory.db lacks a column
    is not acceptable either — the data is the whole point of the app.
    Only ADD COLUMN is attempted; anything else is left to a human.
    """
    inspector = inspect(target)
    added: list[str] = []
    existing_tables = set(inspector.get_table_names())

    with target.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl_type = column.type.compile(dialect=target.dialect)
                # New columns must be nullable or carry a literal default;
                # SQLite cannot add a NOT NULL column without one.
                default = ""
                if column.default is not None and getattr(
                    column.default, "is_scalar", False
                ):
                    value = column.default.arg
                    if isinstance(value, bool):
                        value = int(value)
                    default = f" DEFAULT {value!r}" if isinstance(value, str) else f" DEFAULT {value}"
                conn.exec_driver_sql(
                    f"ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl_type}{default}"
                )
                added.append(f"{table.name}.{column.name}")
    return added


@contextmanager
def get_session() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on error."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def configure_for_testing(database_url: str) -> Engine:
    """Repoint the module-level engine — used by tests for an in-memory DB.

    Lives here rather than in the tests so that the "only this file knows
    the database" rule survives contact with the test suite.
    """
    global engine, SessionFactory
    engine = _make_engine(database_url, echo=False)
    SessionFactory.configure(bind=engine)
    init_db(engine)
    return engine
