"""Every test gets a fresh, empty database. No API key, no network.

In-memory rather than a temp file: creating the schema on disk costs ~1.5s
per test from fsync alone, which turns the suite into something you avoid
running. StaticPool (see session.py) keeps the in-memory database alive
across connections.
"""

from __future__ import annotations

import os

import pytest

# Set before app.config is imported anywhere, so the real memory.db is
# never touched by a test run.
TEST_DATABASE_URL = "sqlite:///:memory:"
os.environ["DATABASE_URL"] = TEST_DATABASE_URL


@pytest.fixture(autouse=True)
def fresh_db():
    from app.db import session as db_session

    db_session.configure_for_testing(TEST_DATABASE_URL)
    yield
    db_session.engine.dispose()
