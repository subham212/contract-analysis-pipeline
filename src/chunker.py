"""
chunker.py
==========
Split long contract text into token-safe chunks.

Why this matters:
  - Contracts often run 30-80 pages (>100k characters).
  - Every LLM has a maximum input size (context window).
  - Even though Gemini 1.5 supports 1M tokens, we still chunk defensively
    so the pipeline works on ANY provider (OpenAI-compat, Llama, etc.).

Strategy: overlap-based sliding window measured in CHARACTERS (a safe
proxy for tokens — ~4 chars/token for English). We keep overlap so that
clauses split across boundaries are not lost.
"""
from __future__ import annotations

from dataclasses import dataclass

from .logger import get_logger

log = get_logger(__name__)

# Rough heuristic: 1 token ≈ 4 characters of English text.
CHARS_PER_TOKEN = 4

# Conservative defaults that fit comfortably inside most models.
DEFAULT_CHUNK_TOKENS = 6_000       # ~24,000 characters
DEFAULT_OVERLAP_TOKENS = 400       # ~1,600 characters of overlap


@dataclass
class Chunk:
    """A slice of the document with positional metadata."""
    index: int          # 0-based chunk number
    text: str           # the chunk content
    char_start: int     # start offset in the original text
    char_end: int       # end offset


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars per token. Good enough for budgeting."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: the full document text.
        chunk_tokens: max tokens per chunk.
        overlap_tokens: tokens of overlap between consecutive chunks.

    Returns:
        List of Chunk objects with positional metadata.
    """
    if not text:
        return []

    chunk_size = chunk_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN

    # If text fits in one chunk, return it as-is
    if len(text) <= chunk_size:
        return [Chunk(index=0, text=text, char_start=0, char_end=len(text))]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        # Try to break at a paragraph or sentence boundary for readability
        boundary = _find_nearest_boundary(text, start, end)
        if boundary > start + overlap:
            end = boundary

        chunk_text_slice = text[start:end].strip()
        if chunk_text_slice:
            chunks.append(
                Chunk(index=idx, text=chunk_text_slice, char_start=start, char_end=end)
            )
            idx += 1

        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end

    log.debug("Chunked %d chars → %d chunks (avg %d chars each)",
              len(text), len(chunks),
              sum(len(c.text) for c in chunks) // max(1, len(chunks)))
    return chunks


def _find_nearest_boundary(text: str, start: int, end: int) -> int:
    """Find the nearest paragraph/sentence boundary at or before `end`."""
    # Prefer paragraph break, then sentence end.
    for sep in ("\n\n", ". ", "? ", "! ", "\n"):
        pos = text.rfind(sep, start, min(end, len(text)))
        if pos > start:
            return pos + len(sep)
    return end


if __name__ == "__main__":
    demo = "Sentence one. Sentence two. " * 500
    parts = chunk_text(demo, chunk_tokens=50, overlap_tokens=10)
    print(f"Demo: {len(demo)} chars → {len(parts)} chunks")
    print(f"First chunk tail: ...{parts[0].text[-60:]!r}")
    print(f"Second chunk head: {parts[1].text[:60]!r}...")
