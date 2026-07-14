"""
llm_client.py
=============
Unified LLM client with automatic provider failover.

Failover chain (per request):
    1. Google Gemini (primary)   ← huge context window, free tier
    2. OpenAI-compatible         ← kicks in if Gemini errors / quota exhausted

Design principles:
  - Single unified interface: generate(prompt, system) -> str
  - Auto-retry within a provider on transient errors.
  - On quota / auth errors, fall through to the next provider.
  - Never leak secrets: keys live only in settings, never logged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .config import settings
from .logger import get_logger

log = get_logger(__name__)

# Errors that mean "this provider is exhausted / misconfigured" → try next
_TRANSIENT_RETRIES = 2
_RETRY_DELAY_SEC = 1.5

# Minimum seconds between calls (to respect free-tier rate limits).
# Set per-provider: free tiers are stricter; paid/fast endpoints can go faster.
_PACING = {
    "gemini": 12.0,        # Gemini free tier: 5 RPM → 12s between calls
    "openai-compat": 1.5,  # NVIDIA NIM free tier: 40 RPM → 1.5s is safe
}


def _min_call_interval(provider_name: str) -> float:
    """Return the minimum seconds between calls for a given provider."""
    return _PACING.get(provider_name, 2.0)


def _extract_retry_seconds(err_msg: str) -> float:
    """
    Parse a 'retry in Ns' hint from a rate-limit error message.
    Returns 0.0 if no hint is found.
    """
    import re
    match = re.search(r"retry[Dd]elay[\"':\s]+(\d+)", err_msg)
    if match:
        return float(match.group(1))
    match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", err_msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"Please retry in (\d+(?:\.\d+)?)", err_msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


class LLMError(Exception):
    """Raised when all providers fail."""


class QuotaExhaustedError(LLMError):
    """Raised when a provider returns a quota / rate-limit error."""


@dataclass
class ProviderResult:
    """Result of a single generation call."""
    text: str
    provider: str
    model: str
    elapsed_sec: float


class GeminiProvider:
    """Google Gemini via the official google-genai SDK (the new SDK)."""

    name = "gemini"
    _client = None

    def __init__(self):
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY not set")
        self.model = settings.gemini_model

    def _ensure_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMError(
                    "google-genai not installed. Run: pip install google-genai"
                ) from exc
            # New SDK: Client auto-reads GEMINI_API_KEY env var,
            # but we pass explicitly for clarity and to match our settings.
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        client = self._ensure_client()
        try:
            from google import genai
            from google.genai import types as gtypes

            # New SDK uses a config object for system instruction + params
            config = gtypes.GenerateContentConfig(
                system_instruction=system if system else None,
                temperature=0.2,
            )
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return resp.text
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("quota", "rate", "429", "resource", "not found", "404")):
                raise QuotaExhaustedError(f"Gemini quota/model error: {exc}") from exc
            raise LLMError(f"Gemini error: {exc}") from exc


class OpenAICompatProvider:
    """OpenAI-compatible provider (works with OpenAI, Groq, Together, OpenRouter, Ollama...)."""

    name = "openai-compat"
    _client = None

    def __init__(self):
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY not set")
        if not settings.openai_model:
            raise LLMError("OPENAI_MODEL not set")
        self.model = settings.openai_model

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("openai not installed. Run: pip install openai") from exc
            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            # OpenRouter requires identification headers per their policy.
            # We add them via default_headers (harmless for other providers).
            base = settings.openai_base_url.lower()
            if "openrouter.ai" in base:
                kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/contract-analysis-pipeline",
                    "X-Title": "Contract Analysis Pipeline",
                }
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("quota", "rate", "429", "insufficient", "temporarily rate-limited")):
                raise QuotaExhaustedError(f"OpenAI-compat quota/rate limit: {exc}") from exc
            raise LLMError(f"OpenAI-compat error: {exc}") from exc


class LLMRouter:
    """
    Routes a generation request through the failover chain.

    Usage:
        router = LLMRouter()
        text = router.generate("Summarize this...", system="You are a lawyer.")
    """

    def __init__(self):
        self.providers: list = []
        self.last_provider: Optional[str] = None
        self._last_call_ts: float = 0.0  # for rate-limit pacing

        # Available provider constructors, keyed by name
        factories = {
            "gemini": lambda: (
                GeminiProvider()
                if settings.gemini_api_key
                else (_ for _ in ()).throw(LLMError("GEMINI_API_KEY not set"))
            ),
            "openai-compat": lambda: (
                OpenAICompatProvider()
                if settings.openai_api_key
                else (_ for _ in ()).throw(LLMError("OPENAI_API_KEY not set"))
            ),
        }

        # Build the chain in the configured order
        order = [p.strip() for p in settings.provider_order.split(",") if p.strip()]
        for name in order:
            if name not in factories:
                log.warning("Unknown provider '%s' in PROVIDER_ORDER, skipping.", name)
                continue
            try:
                provider = factories[name]()
                self.providers.append(provider)
                if name == "gemini":
                    log.info("LLM provider registered: Gemini (%s)", settings.gemini_model)
                else:
                    base = settings.openai_base_url or "api.openai.com"
                    log.info("LLM provider registered: OpenAI-compat (%s @ %s)",
                             settings.openai_model, base)
            except LLMError as exc:
                log.warning("%s not available: %s", name, exc)

        if not self.providers:
            raise LLMError(
                "No LLM provider available. Configure GEMINI_API_KEY and/or "
                "OPENAI_API_KEY in your .env file."
            )

        log.info("Failover chain: %s", " → ".join(p.name for p in self.providers))

    def _pace(self, provider_name: str = "openai-compat") -> None:
        """Sleep just enough to stay under this provider's rate limit."""
        elapsed = time.time() - self._last_call_ts
        wait = _min_call_interval(provider_name) - elapsed
        if wait > 0:
            log.debug("Rate-limit pacing (%s): sleeping %.1fs", provider_name, wait)
            time.sleep(wait)
        self._last_call_ts = time.time()

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_retries_per_provider: int = _TRANSIENT_RETRIES) -> ProviderResult:
        """
        Send a prompt through the failover chain.

        Tries each provider in order. On quota exhaustion, moves to the next.
        On transient errors, retries within the same provider.
        """
        last_error: Optional[Exception] = None

        for provider in self.providers:
            for attempt in range(1, max_retries_per_provider + 1):
                self._pace(provider.name)  # respect this provider's rate limit
                start = time.time()
                try:
                    out = provider.generate(prompt, system=system)
                    elapsed = time.time() - start
                    self.last_provider = provider.name
                    if attempt > 1:
                        log.info("Recovered on %s (attempt %d)", provider.name, attempt)
                    log.debug("LLM call OK: provider=%s model=%s %.2fs %d chars",
                              provider.name, provider.model, elapsed, len(out))
                    return ProviderResult(
                        text=out, provider=provider.name,
                        model=provider.model, elapsed_sec=elapsed
                    )
                except QuotaExhaustedError as exc:
                    # If we're told to retry after N seconds, wait and retry
                    # once on the same provider before falling through.
                    retry_after = _extract_retry_seconds(str(exc))
                    if retry_after and attempt < max_retries_per_provider:
                        log.warning("Rate-limited on %s; waiting %ds then retrying.",
                                    provider.name, retry_after)
                        time.sleep(retry_after + 1)
                        self._last_call_ts = time.time()
                        continue
                    log.warning("Quota exhausted on %s → falling through. (%s)",
                                provider.name, exc)
                    last_error = exc
                    break  # go to next provider
                except LLMError as exc:
                    last_error = exc
                    log.warning("Attempt %d/%d on %s failed: %s",
                                attempt, max_retries_per_provider, provider.name, exc)
                    if attempt < max_retries_per_provider:
                        time.sleep(_RETRY_DELAY_SEC * attempt)
                    continue

        raise LLMError(f"All providers failed. Last error: {last_error}")


# ─── Module-level convenience singleton ──────────────────────────
_router: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    """Get the shared LLM router (lazy-initialized)."""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
