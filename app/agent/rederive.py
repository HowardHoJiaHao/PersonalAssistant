"""Re-derive layer 2 from layer 1.

The central claim of this design is that facts are disposable: because
notes are immutable and verbatim, the whole facts table can be dropped and
rebuilt from the notes after improving the extraction prompt, losing
nothing. This module is what makes that claim testable rather than
aspirational — an untested property is one you find out about on the day
you need it.

Notes are never written here. Only find_person, upsert_person and
save_facts are exposed, so a confused model cannot corrupt layer 1.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, Optional

from app.agent.prompts import build_extraction_prompt
from app.agent.tools import TOOL_SCHEMAS, dispatch
from app.config import settings
from app.db import repository as repo
from app.db.session import get_session
from app.llm.base import LLMProvider
from app.llm.openai_compat import get_provider

# Read tools plus the two writers that touch layer 2 only. save_note is
# deliberately absent: re-derivation must never write a note.
ALLOWED_TOOLS = {"find_person", "upsert_person", "save_facts", "get_person_facts"}

_EXTRACTION_SCHEMAS = [
    schema for schema in TOOL_SCHEMAS if schema["function"]["name"] in ALLOWED_TOOLS
]

OnProgress = Callable[[int, int, str], None]


async def _extract_from_note(
    llm: LLMProvider, owner_id: str, note: dict[str, Any]
) -> int:
    """Run one note through the model. Returns how many facts it saved."""
    with get_session() as session:
        known = [
            f"{p.display_name} (id={p.id}{', ' + p.disambiguator if p.disambiguator else ''})"
            for p in repo.list_people(session, owner_id)
        ]

    user_block = (
        f"note_id: {note['note_id']}\n"
        f"date: {note['event_date'] or 'unknown'}\n"
        f"location: {note['location'] or 'unknown'}\n"
        f"people already known: {', '.join(known) if known else 'none yet'}\n\n"
        f"note text:\n{note['raw_text']}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_extraction_prompt()},
        {"role": "user", "content": user_block},
    ]

    saved = 0
    for _ in range(settings.max_iterations):
        response = await llm.complete(messages, tools=_EXTRACTION_SCHEMAS)
        if not response.wants_tools:
            break

        messages.append(response.raw_message or {"role": "assistant", "content": ""})
        for call in response.tool_calls:
            if call.parse_error or call.name not in ALLOWED_TOOLS:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": call.parse_error or f"{call.name} is not available here",
                }
            else:
                args = dict(call.arguments)
                if call.name == "save_facts":
                    # Force provenance: a re-derived fact with no link back
                    # to its note would break the very property this proves.
                    for item in args.get("facts", []) or []:
                        item["source_note_id"] = note["note_id"]
                result = await asyncio.to_thread(dispatch, call.name, args, owner_id)
                if call.name == "save_facts" and result.get("ok"):
                    saved += sum(
                        1 for r in result.get("saved", []) if r.get("action") != "unchanged"
                    )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result, default=str),
                }
            )
    return saved


async def rederive_facts(
    owner_id: str,
    provider: Optional[LLMProvider] = None,
    on_progress: Optional[OnProgress] = None,
) -> dict[str, Any]:
    """Wipe this owner's facts and rebuild them from their notes.

    Notes are read in chronological order so that supersession replays in
    the order things actually happened — out of order, a 2024 job would
    end up superseding a 2026 one.
    """
    llm = provider or get_provider()

    with get_session() as session:
        notes = sorted(
            (repo.note_to_dict(n) for n in repo.search_notes(session, owner_id, "", limit=100_000)),
            key=lambda n: (n["event_date"] or "0000-00-00", n["note_id"]),
        )
        wiped = repo.delete_facts_for_owner(session, owner_id)

    total, saved, failed = len(notes), 0, 0
    for index, note in enumerate(notes, start=1):
        if on_progress:
            on_progress(index, total, note["raw_text"])
        try:
            saved += await _extract_from_note(llm, owner_id, note)
        except Exception:
            # One bad note must not abandon the rebuild half-done.
            failed += 1

    return {"notes_processed": total, "facts_wiped": wiped, "facts_saved": saved, "failed": failed}
