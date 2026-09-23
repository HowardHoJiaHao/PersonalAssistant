"""Markdown export — one file per person.

A .db file you cannot open is not a backup you trust. These files are
readable in any text editor, greppable, diffable and survive this app
being deleted, which matters for data you cannot recreate.

Notes are included verbatim, because they are the layer that cannot be
regenerated; facts can always be re-derived from them.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from app.db import repository as repo
from app.db.session import get_session


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unnamed"


def _person_markdown(session, owner_id: str, person) -> str:
    facts = repo.get_person_facts(session, owner_id, person.id, include_past=True)
    relations = repo.get_relations(session, owner_id, person.id, include_past=True)
    names = repo.names_for(
        session, owner_id, [r.from_person for r in relations] + [r.to_person for r in relations]
    )
    notes = repo.search_notes(session, owner_id, "", person_id=person.id, limit=1000)

    lines: list[str] = [f"# {person.display_name}", ""]
    meta = [
        f"**{k}:** {v}"
        for k, v in (
            ("relationship", person.relationship),
            ("how we met", person.how_we_met),
            ("also known as", ", ".join(person.aliases or []) or None),
            ("note", person.disambiguator),
            ("mentions", person.mention_count or None),
        )
        if v
    ]
    if meta:
        lines += [" · ".join(meta), ""]

    current = [f for f in facts if f.valid_to is None]
    past = [f for f in facts if f.valid_to is not None]

    if current:
        lines += ["## Current", ""]
        for f in sorted(current, key=lambda x: (x.category, x.key)):
            bits = [f"- **{f.value}** `{f.category}`"]
            if f.reason:
                bits.append(f"— {f.reason}")
            if f.date_value:
                bits.append(f"({f.date_value}{', yearly' if f.recurring else ''})")
            elif f.valid_from:
                bits.append(f"_(since {f.valid_from})_")
            lines.append(" ".join(bits))
        lines.append("")

    if past:
        lines += ["## Previously", ""]
        for f in sorted(past, key=lambda x: (x.category, x.valid_to or date.min)):
            lines.append(
                f"- {f.value} `{f.category}` _({f.valid_from or '?'} → {f.valid_to})_"
            )
        lines.append("")

    if relations:
        lines += ["## Relations", ""]
        for r in relations:
            other = r.to_person if r.from_person == person.id else r.from_person
            ended = f" _(until {r.valid_to})_" if r.valid_to else ""
            lines.append(f"- {r.type}: {names.get(other, '?')}{ended}")
        lines.append("")

    if notes:
        lines += ["## Notes", "", "_Your own words, unedited._", ""]
        for n in sorted(notes, key=lambda x: (x.event_date or date.min, x.id)):
            when = n.event_date or (n.created_at.date() if n.created_at else "?")
            where = f" — {n.location}" if n.location else ""
            lines += [f"**{when}**{where}", "", f"> {n.raw_text}", ""]

    return "\n".join(lines).rstrip() + "\n"


def export_markdown(owner_id: str, out_dir: str | Path) -> dict[str, Any]:
    """Write one markdown file per person, plus an index."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    with get_session() as session:
        people = repo.list_people(session, owner_id)
        for person in people:
            path = out / f"{_slug(person.display_name)}-{person.id}.md"
            path.write_text(_person_markdown(session, owner_id, person), encoding="utf-8")
            written.append(path.name)

        summary = repo.stats(session, owner_id)
        # Notes mentioning nobody would otherwise be exported nowhere.
        orphans = [
            n
            for n in repo.search_notes(session, owner_id, "", limit=1000)
            if not (n.person_ids or [])
        ]
        if orphans:
            body = ["# Unattached notes", "", "_Notes not linked to any person._", ""]
            for n in sorted(orphans, key=lambda x: (x.event_date or date.min, x.id)):
                body += [f"**{n.event_date or n.created_at.date()}**", "", f"> {n.raw_text}", ""]
            (out / "unattached-notes.md").write_text("\n".join(body), encoding="utf-8")
            written.append("unattached-notes.md")

    index = [
        f"# Relationship Memory export",
        "",
        f"_{date.today()} · {summary['people']} people · {summary['notes']} notes · "
        f"{summary['current_facts']} current facts_",
        "",
    ]
    with get_session() as session:
        for person in repo.list_people(session, owner_id):
            index.append(
                f"- [{person.display_name}]({_slug(person.display_name)}-{person.id}.md)"
            )
    if "unattached-notes.md" in written:
        index.append("- [Unattached notes](unattached-notes.md)")
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    return {"ok": True, "directory": str(out.resolve()), "files": len(written) + 1}
