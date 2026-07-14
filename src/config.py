"""
config.py
=========
Central configuration loader.
Reads environment variables from .env and exposes them as typed settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time.
# override=True is CRITICAL: without it, python-dotenv will NOT replace
# environment variables that already exist in the OS env (e.g. a stale key
# exported in an earlier terminal session). This caused a real bug where the
# .env had a fresh OpenRouter key but the code kept reading an old cached key.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ─── Dataset sources (verified, never guess) ─────────────────────
CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_SEED_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/seed.xlsx"


def _get_bool(key: str, default: bool = False) -> bool:
    """Parse a boolean env var."""
    val = os.getenv(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def _get_int(key: str, default: int) -> int:
    """Parse an int env var with fallback."""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """All runtime settings, loaded from environment."""

    # ─── LLM: Primary (Gemini) ───
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    # ─── LLM: Fallback (OpenAI-compatible) ───
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", ""))

    # ─── Provider control ───
    # Comma-separated order of providers to try. Lets you skip a dead provider
    # without removing its key. e.g. "openai-compat" or "gemini,openai-compat"
    provider_order: str = field(
        default_factory=lambda: os.getenv("PROVIDER_ORDER", "gemini,openai-compat")
    )

    # ─── Pipeline ───
    max_contracts: int = field(default_factory=lambda: _get_int("MAX_CONTRACTS", 50))
    contracts_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CONTRACTS_DIR", "data/raw/contracts")
        ).resolve()
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output")).resolve()
    )
    extraction_mode: str = field(
        default_factory=lambda: os.getenv("EXTRACTION_MODE", "zero_shot").lower()
    )
    enable_semantic_search: bool = field(
        default_factory=lambda: _get_bool("ENABLE_SEMANTIC_SEARCH", True)
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ─── Derived ───
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty = OK)."""
        problems: list[str] = []
        if not self.gemini_api_key and not self.openai_api_key:
            problems.append(
                "No LLM API key found. Set GEMINI_API_KEY and/or OPENAI_API_KEY in .env"
            )
        if self.extraction_mode not in ("zero_shot", "few_shot"):
            problems.append(
                f"EXTRACTION_MODE='{self.extraction_mode}' is invalid. "
                "Use 'zero_shot' or 'few_shot'."
            )
        return problems


# Singleton-ish instance
settings = Settings()
