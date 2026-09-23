"""Local Whisper transcription via faster-whisper.

Optional dependency, imported lazily: the model download is large and most
people run this text-only, so a missing package must not stop the app from
starting. `pip install faster-whisper` enables it.

## On code-switched speech

These notes mix English, Malay and Chinese, often inside one sentence
("makan with Peter at the 茶室 semalam"). Whisper is genuinely imperfect
at this, and it is worth being precise about why rather than pretending
otherwise: it detects ONE language per ~30-second window and decodes
conditioned on that choice. Given rojak it picks the dominant language and
tends to push the rest toward it — Malay words nudged into similar-sounding
English, or romanised speech rendered into Hanzi.

Three things are done about it here:

1. `language=None` — auto-detect, never pinned. Pinning a language makes
   Whisper *translate* the other two instead of transcribing them, which
   is worse than a clumsy transcript.
2. `initial_prompt` primes the decoder with a code-switched sample. It
   biases the style and measurably reduces the "correct everything into
   one language" behaviour. Tune it in .env to your own speech.
3. `condition_on_previous_text=False` — stops one bad window from
   dragging the rest of the note with it.

None of that makes it perfect, so the real mitigation is architectural:
the transcript becomes an IMMUTABLE note, so the audio path is stored
alongside it and the interface confirms the text before saving. When the
transcript is wrong, the recording is still the source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.config import settings


class WhisperTranscriber:
    """Implements the Transcriber protocol using faster-whisper."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "auto",
        initial_prompt: str | None = None,
    ) -> None:
        self.model_size = model_size or settings.whisper_model
        self.device = device
        self.initial_prompt = (
            initial_prompt if initial_prompt is not None else settings.whisper_prompt
        )
        self._model: Optional[Any] = None

    def _load(self) -> Any:
        """Load on first use, then keep it — loading takes several seconds."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "faster-whisper is not installed. Run:\n"
                    "  pip install faster-whisper\n"
                    "The model downloads automatically on first use."
                ) from exc
            # int8 keeps memory low enough to run alongside everything else
            # on a small machine; float16 would need a GPU anyway.
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type="int8"
            )
        return self._model

    async def transcribe(
        self, audio_path: str | Path, language: str | None = None
    ) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"no audio file at {path}")

        import asyncio

        def _run() -> str:
            model = self._load()
            segments, _info = model.transcribe(
                str(path),
                # language=None means auto-detect. Do not "helpfully"
                # default it — see the module docstring.
                language=language,
                initial_prompt=self.initial_prompt or None,
                # Each window decoded on its own merits, so a misread
                # sentence cannot drag the rest of the note after it.
                condition_on_previous_text=False,
                beam_size=settings.whisper_beam_size,
                vad_filter=True,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

        # Transcription is CPU-bound and blocking; keep the loop responsive.
        return await asyncio.to_thread(_run)


def get_transcriber() -> Any:
    """Factory mirroring get_provider(). Returns the stub when voice is off."""
    if not settings.voice_enabled:
        from app.transcription.base import NullTranscriber

        return NullTranscriber()
    return WhisperTranscriber()
