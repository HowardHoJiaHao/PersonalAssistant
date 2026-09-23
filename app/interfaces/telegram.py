"""Telegram interface — the seam paying off.

This file contains no memory logic. It maps a Telegram update onto
`run_agent(text, owner_id)` and prints the reply back, exactly as the
terminal does. `owner_id` becomes the Telegram user id, which is why
multi-user needed no schema change: every query has been scoped by
owner_id since the first commit.

Optional dependency:  pip install "python-telegram-bot>=21"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agent.loop import reset_history, run_agent
from app.config import settings
from app.db import repository as repo
from app.db.session import get_session, init_db

logger = logging.getLogger(__name__)

VOICE_DIR = Path("audio")

# Transcripts awaiting confirmation, keyed by telegram user id.
#
# A transcript becomes an IMMUTABLE note, and Whisper mangles
# code-switched speech often enough that saving it unseen would quietly
# corrupt layer 1 — the one layer that is supposed to be exactly what was
# said. So voice notes are shown first and saved only on confirmation.
# In-process like the conversation history; moves to Redis alongside it if
# this ever becomes a long-lived service.
_PENDING: dict[str, dict[str, str]] = {}


def _authorised(user_id: str) -> bool:
    """Allow-list check.

    This bot writes to a private database about real people. An unlisted
    sender is refused rather than quietly given their own tenant, because
    a stranger silently accumulating memory in your app is worse than a
    confusing error message.
    """
    return user_id in settings.telegram_allowed_users


async def _reply_for(
    text: str, user_id: str, source: str = "text", audio_path: str | None = None
) -> str:
    return await run_agent(text, user_id, source=source, audio_path=audio_path)


def build_application() -> Any:
    try:
        from telegram import (
            InlineKeyboardButton,
            InlineKeyboardMarkup,
            Update,
        )
        from telegram.ext import (
            ApplicationBuilder,
            CallbackQueryHandler,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            'python-telegram-bot is not installed. Run:\n'
            '  pip install "python-telegram-bot>=21"'
        ) from exc

    if not settings.telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN is not set in .env")
    if not settings.telegram_allowed_users:
        raise RuntimeError(
            "TELEGRAM_ALLOWED_USERS is empty. Set it to your numeric Telegram user "
            "id — without it, anyone who finds the bot could write to your memory."
        )

    init_db()

    async def guard(update: "Update") -> str | None:
        user_id = str(update.effective_user.id)
        if not _authorised(user_id):
            await update.message.reply_text("Not authorised.")
            logger.warning("rejected telegram user %s", user_id)
            return None
        return user_id

    async def on_text(update: "Update", _ctx: "ContextTypes.DEFAULT_TYPE") -> None:
        user_id = await guard(update)
        if user_id is None:
            return
        text = update.message.text

        # Typing while a transcript is pending is how you correct it: the
        # typed words replace the transcript, and the recording stays
        # attached so the original is never lost.
        pending = _PENDING.pop(user_id, None)
        source, audio_path = "text", None
        if pending:
            source, audio_path = "voice", pending["audio_path"]
            await update.message.reply_text("Using your correction instead.")

        await update.message.chat.send_action("typing")
        try:
            await update.message.reply_text(
                await _reply_for(text, user_id, source=source, audio_path=audio_path)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("telegram turn failed")
            await update.message.reply_text(f"Something went wrong: {type(exc).__name__}")

    async def on_voice(update: "Update", _ctx: "ContextTypes.DEFAULT_TYPE") -> None:
        """Voice note -> transcript -> the same run_agent call."""
        user_id = await guard(update)
        if user_id is None:
            return
        if not settings.voice_enabled:
            await update.message.reply_text(
                "Voice is off. Set VOICE_ENABLED=1 and install faster-whisper."
            )
            return

        from app.transcription.whisper import get_transcriber

        VOICE_DIR.mkdir(exist_ok=True)
        voice = update.message.voice or update.message.audio
        path = VOICE_DIR / f"{user_id}-{voice.file_unique_id}.ogg"
        telegram_file = await voice.get_file()
        await telegram_file.download_to_drive(str(path))

        await update.message.chat.send_action("typing")
        transcript = await get_transcriber().transcribe(path)
        if not transcript:
            await update.message.reply_text("I couldn't make out any speech there.")
            return

        # Confirm before saving. The transcript becomes an immutable note,
        # and mixed English/Malay/Chinese is exactly where Whisper slips.
        _PENDING[user_id] = {"transcript": transcript, "audio_path": str(path)}
        await update.message.reply_text(
            f'Heard: "{transcript}"\n\nSave it, or just type the correction.',
            reply_markup=InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton("Save", callback_data="voice:save"),
                    InlineKeyboardButton("Discard", callback_data="voice:drop"),
                ]]
            ),
        )

    async def on_voice_choice(update: "Update", _ctx: "ContextTypes.DEFAULT_TYPE") -> None:
        query = update.callback_query
        user_id = str(query.from_user.id)
        await query.answer()
        if not _authorised(user_id):
            return

        pending = _PENDING.pop(user_id, None)
        if pending is None:
            await query.edit_message_text("That one's already dealt with.")
            return
        if query.data == "voice:drop":
            # The note is never written; the recording stays on disk.
            await query.edit_message_text("Dropped. The recording is still saved.")
            return

        await query.edit_message_text(f'Saved: "{pending["transcript"]}"')
        await query.message.chat.send_action("typing")
        await query.message.reply_text(
            await _reply_for(
                pending["transcript"],
                user_id,
                source="voice",
                audio_path=pending["audio_path"],
            )
        )

    async def on_new(update: "Update", _ctx: "ContextTypes.DEFAULT_TYPE") -> None:
        user_id = await guard(update)
        if user_id is None:
            return
        reset_history(user_id)
        await update.message.reply_text("New conversation — everything I know is still stored.")

    async def on_people(update: "Update", _ctx: "ContextTypes.DEFAULT_TYPE") -> None:
        user_id = await guard(update)
        if user_id is None:
            return
        with get_session() as session:
            people = repo.list_people(session, user_id)
        await update.message.reply_text(
            "\n".join(f"• {p.display_name}" for p in people) or "Nobody yet."
        )

    app = ApplicationBuilder().token(settings.telegram_token).build()
    app.add_handler(CallbackQueryHandler(on_voice_choice, pattern=r"^voice:"))
    app.add_handler(CommandHandler("new", on_new))
    app.add_handler(CommandHandler("people", on_people))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_application().run_polling()


if __name__ == "__main__":
    main()
