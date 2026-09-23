"""Voice seam — an empty slot for phase 2.

Nothing implements this yet. It exists now so that adding voice is a new
file plus a config line, rather than a change to the agent loop: voice
will call `run_agent(transcript, owner_id)` exactly as the terminal does,
setting the note's `source="voice"` and `audio_path`.

Language note: these notes mix English, Malay and Chinese, often inside
one sentence ("makan with Peter at 茶室 tomorrow"). Forcing a language
code makes Whisper mistranslate the other two, so prefer auto-detect and
let the model handle the code-switching. Keep `language` optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(
        self, audio_path: str | Path, language: str | None = None
    ) -> str:
        """Return the transcript. `language=None` means auto-detect."""
        ...


class NullTranscriber:
    """Placeholder so the seam is importable before voice exists."""

    async def transcribe(
        self, audio_path: str | Path, language: str | None = None
    ) -> str:
        raise NotImplementedError(
            "Voice input is not wired up yet. Phase 2: implement a Transcriber "
            "(faster-whisper locally, or an API), then have the interface pass "
            "the transcript to run_agent() with source='voice'."
        )
