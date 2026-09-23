"""All settings in one place, env-driven.

Nothing else in the app reads os.environ. Adding a deployment target
(Postgres, a different model host) should mean editing .env, not code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    # Treat an empty string in .env the same as unset; otherwise a stray
    # "DATABASE_URL=" line silently produces an unusable config.
    value = os.getenv(name)
    return value if value else default


def _resolve_sqlite_path(url: str) -> str:
    """Anchor a relative SQLite file to the project, not the shell's cwd.

    `sqlite:///memory.db` is cwd-relative, so running from app/ would
    silently open a SECOND, empty database and the memory would look
    like it had vanished. Postgres URLs and :memory: pass through.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    path = url[len(prefix) :]
    if not path or path.startswith(("/", ":")):  # absolute, or :memory:
        return url
    return f"{prefix}{(PROJECT_ROOT / path).resolve()}"


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Database seam -------------------------------------------------
    # Moving to Postgres is a change to this URL and nothing else. The
    # models deliberately avoid dialect-specific column types so the same
    # declarative classes run on both.
    database_url: str = field(
        default_factory=lambda: _resolve_sqlite_path(
            _env("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'memory.db'}")
        )
    )
    sql_echo: bool = field(
        default_factory=lambda: _env("SQL_ECHO", "0").lower() in {"1", "true", "yes"}
    )

    # --- Model seam ----------------------------------------------------
    # base_url alone decides the provider: DeepSeek, Groq, OpenRouter,
    # Together, or a local vLLM/Ollama server. They all speak the same
    # OpenAI-compatible wire format, including function calling.
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com")
    )
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-flash"))
    llm_temperature: float = field(
        default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2"))
    )
    llm_timeout_seconds: int = field(
        default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 60)
    )

    # --- Agent loop ----------------------------------------------------
    # Hard stop on the tool-calling loop. A model that keeps calling tools
    # without ever answering should cost a bounded number of requests.
    max_iterations: int = field(default_factory=lambda: _env_int("MAX_ITERATIONS", 8))
    # Conversation context only. Long-term memory lives in the database,
    # so the window can stay small without losing anything.
    history_turns: int = field(default_factory=lambda: _env_int("HISTORY_TURNS", 10))

    # --- Voice seam ----------------------------------------------------
    # Off by default: faster-whisper is an optional dependency and the
    # model download is large, so text-only users pay nothing for it.
    voice_enabled: bool = field(
        default_factory=lambda: _env("VOICE_ENABLED", "0").lower() in {"1", "true", "yes"}
    )
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "medium"))
    whisper_beam_size: int = field(
        default_factory=lambda: _env_int("WHISPER_BEAM_SIZE", 5)
    )
    # Primes the decoder for code-switched speech. Whisper commits to one
    # language per window, so a sample of how you actually talk measurably
    # reduces it "correcting" Malay and Chinese into English.
    whisper_prompt: str = field(
        default_factory=lambda: _env(
            "WHISPER_PROMPT",
            "Lepak with Wai Keong at the mamak semalam, he cakap he wants to "
            "makan matcha ice cream. 明天 we go badminton at Bangsar lah.",
        )
    )

    # --- Telegram interface ---------------------------------------------
    telegram_token: str = field(default_factory=lambda: _env("TELEGRAM_TOKEN", ""))
    # Empty means anyone who finds the bot can write to your memory, so the
    # interface refuses to start without an explicit allow-list.
    telegram_allowed_users: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            u.strip() for u in _env("TELEGRAM_ALLOWED_USERS", "").split(",") if u.strip()
        )
    )

    # --- Single-user default -------------------------------------------
    # owner_id is threaded through every query from day one, so adding
    # Telegram (owner_id = telegram user id) needs no schema change.
    # Shown on your own profile row. Only cosmetic — the row is found by
    # is_self, never by name.
    owner_name: str = field(default_factory=lambda: _env("OWNER_NAME", "Me"))
    default_owner_id: str = field(
        default_factory=lambda: _env("DEFAULT_OWNER_ID", "local")
    )


settings = Settings()
