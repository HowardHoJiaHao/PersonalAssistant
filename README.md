# Relationship Memory

A terminal chatbot that remembers details about the people in your life.
You talk to it like a diary; it stores your exact words, distils durable
facts, and later answers questions from what you actually said.

```
you › had lunch with Peter at Village Park today, he's really into matcha now,
      got into it in Kyoto last year, he was late again lol
  · save_note(raw_text=had lunch with Peter at Village Park…, event_date=2026-09-23)
  · find_person(name=Peter)
  · upsert_person(display_name=Peter, relationship=friend)
  · save_facts(facts=[{'person_id': 1, 'category': 'food', 'key': 'matcha'…)
› Noted — Peter's big on matcha now.

you › Peter left Maybank, started at Grab on Tuesday
  · save_note(raw_text=Peter left Maybank, started at Grab…, event_date=2026-09-22)
  · save_facts(facts=[{'person_id': 1, 'category': 'work', 'key': 'employer'…)
› Updated — he moved from Maybank to Grab.

you › who's the one into matcha, and why?
  · search_facts(query=matcha)
  · get_fact_provenance(fact_id=1)
› Peter — he got into it in Kyoto last year. You mentioned it over lunch at
  Village Park on 23 Sep.

you › /person peter

Peter
  friend

  now
  • likes matcha a lot [food] — got into it in Kyoto (2026-09-23)
  • works at Grab [work] (2026-09-22)

  previously
  ◦ works at Maybank [work] (2026-01-10 → 2026-09-22)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your DeepSeek key
python main.py
```

Tests need no API key and no network. The whole suite runs in about three
seconds against an in-memory database:

```bash
pytest -q          # or: pytest -q -s   to see the scenario walkthrough
```

**Model names move.** DeepSeek retired `deepseek-chat` and `deepseek-reasoner`.
As of 2026-09-23 the live names are `deepseek-flash` (the default here) and
`deepseek-v4-pro`. If you get a 400 on the first message, check
https://api-docs.deepseek.com/quick_start/pricing and update `LLM_MODEL`.

## The two memory layers

This is the core design decision. The two layers are not redundant.

**`notes` — layer 1.** Your exact words, verbatim. Immutable: never edited,
never deleted. The source of truth.

**`facts` — layer 2.** Distilled claims about one person. Keyed, updatable,
expirable. Lossy by design.

One note usually produces **zero or one** fact. From the Village Park message:

- fact: `food / matcha / "likes matcha a lot"`, reason `"got into it in Kyoto"`
- **not** facts: the lunch, the restaurant, him being late — those live in the note

### Why both exist

Facts answer **what**: *"what does Peter like?"* → `likes matcha a lot`.

Only the raw note answers **when, where, why**: *"when and why did Peter get
into matcha?"* → *"over lunch at Village Park on 23 Sep — he got into it in
Kyoto last year."*

That is why every fact carries a `source_note_id` pointing back at the note it
came from. That one foreign key is what makes provenance questions answerable
at all.

### Two consequences worth protecting

**Facts are disposable; notes are not.** Because notes are immutable, you can
wipe the entire `facts` table and re-derive it from `notes` after improving the
extraction prompt, and lose nothing. `repository.delete_facts_for_owner()`
exists for exactly this. Don't add anything that makes a fact the only copy of
a detail — that breaks the property.

**Facts are superseded, never overwritten.** When Peter moves from Maybank to
Grab, the old row gets `valid_to` and `superseded_by` rather than being
updated. So *"where does Peter work?"* says Grab, and *"where did he work
before?"* still says Maybank, with an end date.

**Fact dates follow the note's `event_date`, not the wall clock.** Write up
Tuesday's lunch on Thursday and *"when did he start at Grab?"* still answers
Tuesday.

## Layout

```
main.py                     entry point
app/
  config.py                 all settings, env-driven
  export.py                 markdown export, one file per person
  llm/
    base.py                 LLMProvider protocol      <- model seam
    openai_compat.py        DeepSeek/Groq/OpenRouter/Together/Ollama/OpenAI
  db/
    models.py               people · notes · facts · relations
    session.py              engine, migrations, FTS5  <- database seam
    repository.py           ALL SQL lives here
  agent/
    prompts.py              system prompt + extraction prompt
    tools.py                tool schemas + dispatcher
    loop.py                 run_agent(text, owner_id)  <- interface seam
    rederive.py             rebuild facts from notes
  interfaces/
    cli.py                  terminal
    telegram.py             Telegram bot
  transcription/
    base.py                 Transcriber protocol      <- voice seam
    whisper.py              faster-whisper (optional)
tests/
  test_backend.py           db + tools scenario
  test_loop.py              agent loop, scripted fake model
  test_key_consistency.py   supersession key drift
  test_search_and_matching.py  FTS5 + fuzzy person matching
  test_features.py          brief, reminders, re-derive, export
```

### Commands

| command | what it does |
|---|---|
| `/me` | what it knows about you |
| `/people` | everyone it knows |
| `/person <name>` | full profile: current facts, past facts with dates, relations, notes |
| `/brief <name>` | what to know before seeing them — facts, recent changes, your own words |
| `/remind` | birthdays and anniversaries due, plus people who have gone quiet |
| `/search <text>` | full-text search across notes and facts |
| `/export [dir]` | markdown, one file per person |
| `/rederive` | wipe facts and rebuild them from the notes |
| `/voice <file>` | transcribe an audio file and file it as a note |
| `/stats` · `/new` · `/quit` | counts · clear conversation (keeps memory) · exit |

Reminders also print at startup, so the app speaks before it is spoken to.

## The three seams

The seams matter more than the features. Each future phase should be a change
in one place.

### 1. Model seam — `app/llm/base.py`

`LLMProvider` has one method: `async complete(messages, tools, model)
-> LLMResponse`. Everything above it speaks `LLMResponse` and `ToolCall` only —
no provider-shaped field reaches the agent loop or the tools.

**To switch provider:** change `LLM_BASE_URL` and `LLM_MODEL` in `.env`. Groq,
OpenRouter, Together, Fireworks, vLLM and Ollama are all OpenAI-compatible, so
`OpenAICompatibleProvider` already covers them. For a provider that isn't
(Anthropic's native API, say), write a class with that one method and add an
elif to `get_provider()`.

Malformed tool-call JSON is returned to the model as a tool error rather than
raising, because smaller models emit bad arguments often enough to matter.

### 2. Database seam — `app/db/session.py`

The only file that knows which database is in use. List columns are `JSON`,
not Postgres `ARRAY`, so the same models run on both.

**To move to Postgres:** change `DATABASE_URL`, `pip install psycopg[binary]`.
Nothing else. No model edits, no repository edits.

### 3. Interface seam — `app/agent/loop.py`

`run_agent(text, owner_id) -> str` knows nothing about terminals.

**Telegram** is [app/interfaces/telegram.py](app/interfaces/telegram.py) and
contains no memory logic — it maps an update onto `run_agent(text,
str(user_id))`. That is the seam paying off: multi-user needed no schema change,
because every query has been scoped by `owner_id` since the first commit.

```bash
pip install "python-telegram-bot>=21"
# .env: TELEGRAM_TOKEN from @BotFather, TELEGRAM_ALLOWED_USERS from @userinfobot
python -m app.interfaces.telegram
```

It refuses to start with an empty allow-list. This bot writes to a private
database about real people; a stranger who finds it should get an error, not
their own tenant.

**Voice** is [app/transcription/whisper.py](app/transcription/whisper.py), an
optional dependency loaded lazily so text-only users never pay for it.

```bash
pip install faster-whisper     # model downloads on first use
# .env: VOICE_ENABLED=1
```

### Voice and code-switched speech — read this before trusting it

These notes mix English, Malay and Chinese inside a single sentence
(*"lepak with Wai Keong at the mamak semalam"*). **Whisper is imperfect at
this, and it is worth knowing why.** It detects one language per ~30-second
window and decodes conditioned on that choice, so given rojak it picks the
dominant language and pulls the rest toward it — Malay words nudged into
similar-sounding English, romanised speech rendered into Hanzi.

Three settings help:

| setting | why |
|---|---|
| `language=None` | Never pinned. Pinning makes Whisper *translate* the other two languages instead of transcribing them — worse than a clumsy transcript. |
| `WHISPER_PROMPT` | Primes the decoder with a sample of code-switched speech. **Rewrite it in your own speaking style** — real names, real mix. |
| `WHISPER_MODEL` | `small` is weak on rojak. `medium` is the sweet spot on 4 cores. `large-v3` is best but wants ~3GB spare RAM. |

None of that makes it reliable, so the real mitigation is architectural. A
transcript becomes an **immutable** note, so:

- the recording path is stored on the note (`source="voice"`, `audio_path`),
  making the audio the fallback source of truth when the transcript is wrong;
- **Telegram confirms before saving** — it shows the transcript with Save /
  Discard, and typing instead replaces it with your correction while keeping
  the recording attached.

Saving an unreviewed transcript would quietly corrupt layer 1, which is the
one thing this design cannot tolerate.

### Still open

- `search_notes` on **Postgres** falls back to `LIKE`. Swap in tsvector, and add
  pgvector embeddings for genuinely semantic recall ("who did I talk to about
  career stuff") — behind the same signature.
- `loop.py` history is an in-process dict. Move it to a table or Redis when this
  becomes a long-lived service.

## Features worth knowing about

### It remembers you, not just everyone else

A personal assistant that knows nothing about the person using it can only
look things up. So the owner gets an **ordinary `Person` row** flagged
`is_self`, rather than a table of their own — which means facts, relations,
notes, search and supersession all work on you with no separate machinery,
and *"my sister is Mei"* is a normal relation instead of a special case.

```
you › my name is Howard, I'm allergic to prawns and I don't drink
you › waikeong loves seafood especially prawns, he's my friend from school
you › I want to bring waikeong out for dinner, where should we go?

› Wai Keong loves seafood, prawns especially — but you're allergic to prawns.
  A mixed seafood place rather than a prawn specialist, and mention the allergy
  when you book; shellfish kitchens cross-contaminate. Skip the wine list as
  the centrepiece too, since you don't drink.
```

That answer needs both halves of the memory, which is the point.

The risk this creates is mis-routing. *"I'm allergic to prawns"* filed against
whoever was mentioned last is both a lost fact about you **and** a false one
about them — and it is the kind that gets acted on at a dinner table. So `I`,
`me`, `my` and `myself` resolve to your own row in `find_person`, the prompt
routes first-person statements to `remember_about_me`, and the model is told to
read `get_about_me` before any advice that depends on your situation.
[tests/test_self.py](tests/test_self.py) pins the routing.

`/people` deliberately excludes you — "who do I know" is a question about other
people — but `search_facts` includes you, so *"who likes durian?"* can answer
"you do".

### The brief — the reason to keep the notes layer

`/brief peter`, or just *"I'm seeing Peter tomorrow"*, returns his current facts
grouped by category, what changed recently (both what started **and** what
ended), how long since you last saw him, and his recent notes **verbatim**.

The model reads those raw notes for loose ends you never resolved — *"in June
you wrote he was interviewing somewhere; did that come through?"*. That question
is unanswerable from the facts table alone. It is the clearest payoff of keeping
layer 1.

Spotting an open thread is a judgement call, so the database does retrieval and
the model does the judging. A keyword heuristic in SQL would be brittle and
would hide its own mistakes.

### Reminders — the app speaking first

Facts carry a `date_value` (and `recurring` for anything annual), so a birthday
is a queryable date rather than prose buried in a note. `/remind` shows what is
due in the next 30 days, plus people you haven't mentioned in 90.

Someone mentioned only once never appears in the out-of-touch list — a passing
acquaintance should not generate guilt forever.

### Re-derivation — the design claim, made testable

`/rederive` deletes every fact and rebuilds them from the notes with the current
extraction prompt. Improve the prompt, re-derive, lose nothing.

This is the property the two-layer design is *for*, and an untested property is
one you discover is broken on the day you need it. The pass can only reach
`find_person`, `upsert_person`, `save_facts` and `get_person_facts` — `save_note`
is not in its tool list at all, so it cannot corrupt layer 1 even if the model
tries. Notes replay in `event_date` order, because out of order a 2024 job would
supersede a 2026 one.

### Export — a backup you can actually read

`/export` writes one markdown file per person: facts, superseded facts with
their date ranges, relations, and every note verbatim. A `.db` file you cannot
open is not a backup you trust.

### Search

SQLite FTS5 with Porter stemming, so *"running"* finds a note that says *"run"*
and *"matcha Kyoto"* finds a note where those words are ten words apart. The
`unicode61` tokenizer keeps Malay and Chinese notes searchable. It falls back to
`LIKE` on Postgres or a SQLite build without FTS5 — a search feature must never
stop the app from starting.

The index is external-content: the notes table remains the only copy of the
words, so the index is derived data that can be dropped and rebuilt, exactly
like facts.

## Safety rules baked into the design

**`owner_id` is injected by the harness, never a tool parameter.** It is the
tenant boundary. If the model could name whose data to read, a confused or
manipulated model could cross it. No tool schema declares it, and `dispatch()`
strips it from arguments if a model invents it. Both test files assert this.

**`find_person` returns a LIST, never a single best guess.** A fact attached to
the wrong Peter is the one error you cannot spot by reading your own data later —
it looks perfectly normal, just filed under the wrong human. When more than one
match comes back, the result carries `ambiguous: true` and the prompt requires
asking rather than guessing.

**There is no `delete_person` tool.** A tool that doesn't exist can't be
misused. Duplicates are handled by `merge_people`, which relocates every row
rather than dropping any.

**Fact keys are normalised, aliased and fuzzy-matched before supersession.**
Supersession matches on `(person_id, key)`, and `key` is a free-text string the
model invents. If it writes `employer` in March and `workplace` in June, nothing
supersedes and **both** facts stay current — the app then reports two jobs held
at once, silently, in data that looks completely normal. `resolve_key()` collapses
known aliases and catches spelling drift above a similarity of 90, a threshold
measured to sit between real drift (`employer`/`employers`, 94) and genuinely
different keys (`matcha`/`mocha`, 73). Resolution is scoped to one person and one
category, so a wrong match can never reach across people.
[tests/test_key_consistency.py](tests/test_key_consistency.py) guards both
directions: drift must merge, and near-misses must not.

**Never invent a detail.** The prompt is explicit that an empty answer is fine
and a wrong one is not — you will act on these answers in front of a real
person.

## A note on what this actually is

This is a database about other people, none of whom consented to it.

Keep it local. Back it up encrypted — `memory.db` and `.env` are both in
`.gitignore`, and they should stay there. Delete it if you stop using it.

The useful test for any note: **would you be comfortable if that person read
it?** Write notes that pass. The system is built so that what you write is
preserved exactly, which cuts both ways.
