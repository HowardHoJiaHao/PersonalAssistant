"""Terminal interface — deliberately thin.

Read input, call run_agent, print the reply. Everything interesting lives
behind the seam; this file is replaceable by a Telegram handler that does
the same three things with the same function.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.agent.loop import reset_history, run_agent
from app.agent.rederive import rederive_facts
from app.export import export_markdown
from app.config import settings
from app.db import repository as repo
from app.db.session import get_session, init_db

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

BANNER = f"""{BOLD}Relationship Memory{RESET}
{DIM}/me  /people  /person <name>  /brief <name>  /remind  /search <text>
/export [dir]  /rederive  /voice <file>  /stats  /new  /quit{RESET}
"""


def _trace(name: str, args: dict[str, Any]) -> None:
    """Dim one-liner so the user can see what the agent is doing.

    Display only — it must never change the loop's behaviour.
    """
    summary = ", ".join(
        f"{k}={str(v)[:42]}" for k, v in args.items() if v not in (None, [], "")
    )
    print(f"{DIM}  · {name}({summary}){RESET}")


def _print_people(owner_id: str) -> None:
    with get_session() as session:
        people = repo.list_people(session, owner_id)
    if not people:
        print(f"{DIM}  nobody yet — tell me about someone{RESET}")
        return
    for p in people:
        tag = f" {DIM}({p.disambiguator}){RESET}" if p.disambiguator else ""
        rel = f" {DIM}· {p.relationship}{RESET}" if p.relationship else ""
        print(f"  {BOLD}{p.display_name}{RESET}{tag}{rel} {DIM}· {p.mention_count} mentions{RESET}")


def _print_profile(owner_id: str, name: str) -> None:
    """Full profile: current facts, past facts with dates, relations."""
    with get_session() as session:
        matches, suggestions = repo.find_people(session, owner_id, name)
        if not matches:
            print(f"{DIM}  no one matching {name!r}{RESET}")
            for person, score in suggestions:
                print(f"{DIM}    did you mean {person.display_name}? ({score}% similar){RESET}")
            return
        if len(matches) > 1:
            # Same rule as the tool: show them all, let the human choose.
            print(f"{YELLOW}  {len(matches)} matches — be more specific:{RESET}")
            for p in matches:
                print(f"    {p.display_name} {DIM}({p.disambiguator or 'no tag'}){RESET}")
            return

        person = matches[0]
        facts = repo.get_person_facts(session, owner_id, person.id, include_past=True)
        relations = repo.get_relations(session, owner_id, person.id, include_past=True)
        names = repo.names_for(
            session,
            owner_id,
            [r.from_person for r in relations] + [r.to_person for r in relations] + [person.id],
        )
        notes = repo.search_notes(session, owner_id, "", person_id=person.id, limit=3)

        print(f"\n{BOLD}{person.display_name}{RESET}", end="")
        if person.disambiguator:
            print(f" {DIM}({person.disambiguator}){RESET}", end="")
        print()
        if person.relationship:
            print(f"  {DIM}{person.relationship}{RESET}", end="")
            if person.how_we_met:
                print(f" {DIM}· met: {person.how_we_met}{RESET}", end="")
            print()

        current = [f for f in facts if f.valid_to is None]
        past = [f for f in facts if f.valid_to is not None]

        if current:
            print(f"\n  {BOLD}now{RESET}")
            for f in current:
                line = f"  {GREEN}•{RESET} {f.value} {DIM}[{f.category}]{RESET}"
                if f.reason:
                    line += f" {DIM}— {f.reason}{RESET}"
                if f.valid_from:
                    line += f" {DIM}({f.valid_from}){RESET}"
                print(line)
        if past:
            print(f"\n  {BOLD}previously{RESET}")
            for f in past:
                print(
                    f"  {DIM}◦ {f.value} [{f.category}] "
                    f"({f.valid_from or '?'} → {f.valid_to}){RESET}"
                )
        if relations:
            print(f"\n  {BOLD}relations{RESET}")
            for r in relations:
                # Phrased from this person's side. Printing the raw
                # from/to row here said "Almond's child is Wai Keong".
                view = repo.relation_from_perspective(r, person.id, names)
                ended = f" {DIM}(until {view['valid_to']}){RESET}" if view["valid_to"] else ""
                print(f"  {DIM}·{RESET} {view['relation']}: {view['other_name']}{ended}")
        if notes:
            print(f"\n  {BOLD}recent notes{RESET}")
            for n in notes:
                print(f"  {DIM}{n.event_date or n.created_at.date()}: {n.raw_text[:88]}{RESET}")
        print()


def _print_reminders(owner_id: str, quiet: bool = False) -> None:
    """Upcoming dates and people who have gone quiet.

    Printed at startup too — a memory that only answers when asked is half
    a memory. `quiet` suppresses the "nothing" message for that case.
    """
    with get_session() as session:
        upcoming = repo.get_upcoming(session, owner_id, within_days=30)
        planned = repo.get_planned_notes(session, owner_id, within_days=30)
        stale = repo.get_neglected(session, owner_id, quiet_days=90)

    if not upcoming and not planned and not stale:
        if not quiet:
            print(f"{DIM}  nothing coming up, nobody overdue{RESET}")
        return

    for item in upcoming:
        when = "today" if item["days_away"] == 0 else f"in {item['days_away']}d"
        print(f"  {YELLOW}◆{RESET} {item['person_name']}: {item['value']} {DIM}({when}){RESET}")
    for note in planned:
        when = "tomorrow" if note["days_away"] == 1 else f"in {note['days_away']}d"
        who = f"{', '.join(note['people'])}: " if note["people"] else ""
        print(f"  {YELLOW}▸{RESET} {who}{note['raw_text'][:66]} {DIM}({when}){RESET}")
    for person in stale:
        months = person["days_quiet"] // 30
        print(
            f"  {DIM}◇ {person['display_name']} — nothing since "
            f"{person['last_mentioned_date']} ({months} months){RESET}"
        )


def _print_brief(owner_id: str, name: str) -> None:
    with get_session() as session:
        matches, _ = repo.find_people(session, owner_id, name)
        if not matches:
            print(f"{DIM}  no one matching {name!r}{RESET}")
            return
        if len(matches) > 1:
            print(f"{YELLOW}  {len(matches)} matches — be more specific:{RESET}")
            for person in matches:
                print(f"    {person.display_name} {DIM}({person.disambiguator or 'no tag'}){RESET}")
            return
        brief = repo.get_person_brief(session, owner_id, matches[0].id)

    print(f"\n{BOLD}{brief['person']['display_name']}{RESET}", end="")
    if brief["days_since_last_seen"] is not None:
        print(f" {DIM}· last seen {brief['days_since_last_seen']}d ago{RESET}", end="")
    print()

    for category, facts in sorted(brief["facts_by_category"].items()):
        print(f"\n  {BOLD}{category}{RESET}")
        for fact in facts:
            reason = f" {DIM}— {fact['reason']}{RESET}" if fact["reason"] else ""
            print(f"  {GREEN}•{RESET} {fact['value']}{reason}")

    if brief["relations"]:
        print(f"\n  {BOLD}people{RESET}")
        for r in brief["relations"]:
            print(f"  {GREEN}•{RESET} {r['relation']}: {r['other_name']}")

    if brief.get("planned"):
        print(f"\n  {BOLD}coming up{RESET}")
        for note in brief["planned"]:
            print(f"  {YELLOW}▸{RESET} {note['event_date']}: {note['raw_text'][:78]}")

    changed = brief["changed_recently"]
    if changed:
        print(f"\n  {BOLD}changed recently{RESET}")
        for fact in changed:
            if fact["is_current"]:
                print(f"  {GREEN}+{RESET} {fact['value']} {DIM}(since {fact['valid_from']}){RESET}")
            else:
                print(f"  {DIM}- {fact['value']} (ended {fact['valid_to']}){RESET}")

    if brief["recent_notes"]:
        print(f"\n  {BOLD}your own words{RESET}")
        for note in brief["recent_notes"][:4]:
            print(f"  {DIM}{note['event_date'] or '?'}: {note['raw_text'][:96]}{RESET}")
    print(f"\n{DIM}  ask me \"what should I follow up with them about?\" for loose ends{RESET}\n")


def _print_search(owner_id: str, query: str) -> None:
    with get_session() as session:
        notes = repo.search_notes(session, owner_id, query, limit=8)
        facts = repo.search_facts(session, owner_id, query, limit=8)
    if facts:
        print(f"  {BOLD}facts{RESET}")
        for fact, person in facts:
            print(f"  {GREEN}•{RESET} {person.display_name}: {fact.value}")
    if notes:
        print(f"  {BOLD}notes{RESET}")
        for note in notes:
            print(f"  {DIM}{note.event_date or '?'}: {note.raw_text[:92]}{RESET}")
    if not notes and not facts:
        print(f"{DIM}  nothing for {query!r}{RESET}")


def _do_export(owner_id: str, target: str) -> None:
    result = export_markdown(owner_id, target or "export")
    print(f"  wrote {result['files']} files to {result['directory']}")
    print(f"{DIM}  readable without this app — that is the point of a backup{RESET}")


async def _do_rederive(owner_id: str) -> None:
    """Wipe facts and rebuild them from the notes.

    Confirmed explicitly because it costs API calls and takes a while —
    though it is safe by construction: notes are untouched, and facts are
    derived data.
    """
    with get_session() as session:
        summary = repo.stats(session, owner_id)
    print(
        f"{YELLOW}  This deletes all {summary['current_facts'] + summary['past_facts']} facts "
        f"and re-extracts them from your {summary['notes']} notes.{RESET}"
    )
    print(f"{DIM}  Your notes are never touched. Costs one API call per note.{RESET}")
    if input("  type 'yes' to continue: ").strip().lower() != "yes":
        print(f"{DIM}  cancelled{RESET}")
        return

    def progress(index: int, total: int, text: str) -> None:
        print(f"{DIM}  [{index}/{total}] {text[:70]}{RESET}")

    result = await rederive_facts(owner_id, on_progress=progress)
    print(
        f"  rebuilt: {result['facts_saved']} facts from {result['notes_processed']} notes "
        f"(wiped {result['facts_wiped']}, {result['failed']} failed)"
    )


async def _do_voice(owner_id: str, path: str) -> None:
    if not settings.voice_enabled:
        print(f"{DIM}  voice is off — set VOICE_ENABLED=1 and pip install faster-whisper{RESET}")
        return
    from app.transcription.whisper import get_transcriber

    print(f"{DIM}  transcribing (auto-detecting language)…{RESET}")
    transcript = await get_transcriber().transcribe(path)
    if not transcript:
        print(f"{DIM}  no speech found{RESET}")
        return
    # Show what was captured: this text becomes an immutable note.
    print(f'  {DIM}heard:{RESET} "{transcript}"')
    reply = await run_agent(
        transcript, owner_id, on_tool=_trace, source="voice", audio_path=str(path)
    )
    print(f"{GREEN}›{RESET} {reply}\n")


def _print_me(owner_id: str) -> None:
    """What the assistant knows about you."""
    with get_session() as session:
        me = repo.get_self(session, owner_id)
        if me is None:
            print(f"{DIM}  nothing about you yet — try \"I'm allergic to prawns\"{RESET}")
            return
        facts = repo.get_person_facts(session, owner_id, me.id, include_past=True)
        relations = repo.get_relations(session, owner_id, me.id)
        names = repo.names_for(
            session, owner_id,
            [r.from_person for r in relations] + [r.to_person for r in relations] + [me.id],
        )

    print(f"\n{BOLD}{me.display_name}{RESET} {DIM}(you){RESET}")
    current = [f for f in facts if f.valid_to is None]
    past = [f for f in facts if f.valid_to is not None]
    for fact in sorted(current, key=lambda f: (f.category, f.key)):
        reason = f" {DIM}— {fact.reason}{RESET}" if fact.reason else ""
        print(f"  {GREEN}•{RESET} {fact.value} {DIM}[{fact.category}]{RESET}{reason}")
    if past:
        print(f"\n  {BOLD}previously{RESET}")
        for fact in past:
            print(f"  {DIM}◦ {fact.value} ({fact.valid_from or '?'} → {fact.valid_to}){RESET}")
    if relations:
        print(f"\n  {BOLD}your people{RESET}")
        for rel in relations:
            view = repo.relation_from_perspective(rel, me.id, names)
            print(f"  {DIM}·{RESET} {view['relation']}: {view['other_name']}")
    print()


def _print_stats(owner_id: str) -> None:
    with get_session() as session:
        s = repo.stats(session, owner_id)
    print(
        f"  {s['people']} people · {s['notes']} notes · "
        f"{s['current_facts']} current facts · {s['past_facts']} past · "
        f"{s['relations']} relations"
    )


def _handle_command(line: str, owner_id: str) -> bool:
    """Return True if the line was a command. /quit raises SystemExit."""
    cmd, _, arg = line[1:].partition(" ")
    cmd, arg = cmd.strip().lower(), arg.strip()

    if cmd in {"quit", "q", "exit"}:
        raise SystemExit(0)
    if cmd == "people":
        _print_people(owner_id)
    elif cmd == "brief":
        if arg:
            _print_brief(owner_id, arg)
        else:
            print(f"{DIM}  usage: /brief <name>{RESET}")
    elif cmd == "me":
        _print_me(owner_id)
    elif cmd in {"remind", "reminders", "upcoming"}:
        _print_reminders(owner_id)
    elif cmd == "search":
        if arg:
            _print_search(owner_id, arg)
        else:
            print(f"{DIM}  usage: /search <text>{RESET}")
    elif cmd == "export":
        _do_export(owner_id, arg)
    elif cmd == "person":
        if arg:
            _print_profile(owner_id, arg)
        else:
            print(f"{DIM}  usage: /person <name>{RESET}")
    elif cmd == "stats":
        _print_stats(owner_id)
    elif cmd == "new":
        # Clears the conversation window only; the database is memory.
        reset_history(owner_id)
        print(f"{DIM}  new conversation — everything I know is still stored{RESET}")
    elif cmd in {"rederive", "voice"}:
        # Async commands are handled by the caller, which owns the loop.
        return False
    else:
        print(f"{DIM}  unknown command — see the banner for the list{RESET}")
    return True


async def chat(owner_id: str | None = None) -> None:
    owner = owner_id or settings.default_owner_id
    init_db()
    print(BANNER)
    # Speak first: surface what is due before being asked.
    _print_reminders(owner, quiet=True)

    while True:
        try:
            line = input(f"{BOLD}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd, arg = cmd.strip().lower(), arg.strip()
            try:
                if cmd == "rederive":
                    await _do_rederive(owner)
                elif cmd == "voice":
                    if arg:
                        await _do_voice(owner, arg)
                    else:
                        print(f"{DIM}  usage: /voice <audio-file>{RESET}")
                else:
                    _handle_command(line, owner)
            except SystemExit:
                return
            except Exception as exc:  # noqa: BLE001
                print(f"{YELLOW}  {type(exc).__name__}: {exc}{RESET}")
            continue

        try:
            reply = await run_agent(line, owner, on_tool=_trace)
            print(f"{GREEN}›{RESET} {reply}\n")
        except KeyboardInterrupt:
            print(f"\n{DIM}  cancelled{RESET}\n")
        except Exception as exc:  # noqa: BLE001
            # Caught per turn: one bad response must not end the session
            # and lose the conversation.
            print(f"{YELLOW}  something went wrong: {type(exc).__name__}: {exc}{RESET}\n")


def main() -> None:
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass
