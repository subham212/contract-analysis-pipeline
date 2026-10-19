"""
summarizer.py
=============
Generates search-optimized summaries for legal section chunks.
Uses persistent JSON disk cache to avoid redundant API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.config import settings
from src.llm_client import LLMError, get_router
from src.logger import get_logger
from src.rag.section_chunker import SectionChunk

log = get_logger(__name__)

_DEFAULT_CACHE_PATH = Path(settings.output_dir) / "rag_summaries_cache.json"

_SYSTEM_PROMPT = (
    "You are a legal contract search indexer. "
    "Write concise search summaries capturing core rights, obligations, notice windows, and penalties. "
    "Never hallucinate. Return only plain text."
)

_SECTION_SUMMARY_PROMPT = """\
Write a dense 2-3 sentence search summary for this contract section.
Highlight operative rights, obligations, specific notice periods, and liability/penalties if present.

CONTRACT: {contract_title}
SECTION: {section_title}
TEXT:
<<<
{raw_text}
>>>

Summary:"""


class SummaryCache:
    """Simple JSON-backed disk cache for chunk summaries."""

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or _DEFAULT_CACHE_PATH
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                self._data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Could not read summary cache: %s", exc)
                self._data = {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, chunk_id: str) -> Optional[str]:
        return self._data.get(chunk_id)

    def set(self, chunk_id: str, summary: str) -> None:
        self._data[chunk_id] = summary


def summarize_chunk(
    chunk: SectionChunk,
    router=None,
    cache: Optional[SummaryCache] = None,
) -> str:
    """
    Produce a 2-3 sentence search summary for a SectionChunk.
    Uses disk cache first to prevent redundant LLM calls.
    """
    # ponytail: don't call an LLM for tiny snippets
    clean_text = chunk.raw_text.strip()
    if len(clean_text) <= 200:
        return clean_text

    # 1. Check cache
    if cache is not None:
        cached = cache.get(chunk.chunk_id)
        if cached:
            return cached

    # 2. Call LLM router
    if router is None:
        router = get_router()

    prompt = _SECTION_SUMMARY_PROMPT.format(
        contract_title=chunk.contract_title,
        section_title=chunk.section_title,
        # Truncate to ~4000 chars for the summary call if the section is very long
        raw_text=clean_text[:4000],
    )

    try:
        res = router.generate(prompt, system=_SYSTEM_PROMPT)
        summary = res.text.strip()
    except LLMError as exc:
        log.warning("LLM summarization failed for %s, falling back to prefix: %s", chunk.chunk_id, exc)
        # ponytail: fallback to clean head prefix if LLM drops
        summary = f"{chunk.section_title}: {clean_text[:250]}..."

    # 3. Store in cache
    if cache is not None:
        cache.set(chunk.chunk_id, summary)

    return summary


def summarize_chunks_batch(
    chunks: list[SectionChunk],
    router=None,
    cache_path: Optional[Path] = None,
) -> dict[str, str]:
    """
    Summarize a list of chunks, saving results to disk cache.
    Returns a dict mapping chunk_id -> summary string.
    """
    cache = SummaryCache(cache_path)
    if router is None:
        router = get_router()

    results: dict[str, str] = {}
    new_summaries = 0

    for chunk in chunks:
        if cache.get(chunk.chunk_id):
            results[chunk.chunk_id] = cache.get(chunk.chunk_id)
            continue

        summary = summarize_chunk(chunk, router=router, cache=cache)
        results[chunk.chunk_id] = summary
        new_summaries += 1

    if new_summaries > 0:
        cache.save()
        log.info("Saved %d new chunk summaries to cache (%s)", new_summaries, cache.cache_path)

    return results


if __name__ == "__main__":
    # Ponytail verification self-check with dummy router
    class MockRouter:
        def generate(self, prompt: str, system: Optional[str] = None):
            from types import SimpleNamespace
            return SimpleNamespace(text="Mock summary: Party A grants rights to Party B.", provider="mock")

    sample_chunk = SectionChunk(
        chunk_id="test_chunk_01",
        contract_title="TestCo_Agreement",
        section_title="1. GRANT OF LICENSE",
        raw_text=(
            "The Licensor hereby grants to the Licensee an exclusive, transferable license to use, reproduce, "
            "and distribute the Software in North America for a period of five (5) years commencing on the Effective Date, "
            "subject to timely payment of annual royalties and compliance with all terms specified herein."
        ),
        char_start=0,
        char_end=280,
    )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cache = Path(tmp_dir) / "test_cache.json"
        cache = SummaryCache(tmp_cache)
        
        # First call generates
        s1 = summarize_chunk(sample_chunk, router=MockRouter(), cache=cache)
        cache.save()
        assert s1 == "Mock summary: Party A grants rights to Party B."
        assert tmp_cache.exists()

        # Second call hits cache (even without router)
        s2 = summarize_chunk(sample_chunk, router=None, cache=cache)
        assert s2 == s1, "Cache hit failed"
        print("Self-check passed: Summarizer + Cache verified successfully!")
