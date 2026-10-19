"""
section_chunker.py
==================
Legal-aware contract chunker that partitions documents by Article/Section headings.
Falls back to paragraph sliding window if text lacks explicit headings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Regex targeting standard legal section and article headings
_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:"
    r"(?:ARTICLE|SECTION|CLAUSE|EXHIBIT|SCHEDULE)\s+[0-9A-ZIVXLCDM\.]+[^\r\n]*|"
    r"\b[0-9]{1,2}\.[ \t]+[A-Z][^\r\n]{1,80}"
    r")",
    re.IGNORECASE,
)

# Target chunk size for search embeddings (~350-500 tokens)
TARGET_CHUNK_CHARS = 1800
MAX_CHUNK_CHARS = 2500
CHUNK_OVERLAP_CHARS = 200


@dataclass
class SectionChunk:
    """A legal document section with title and text."""
    chunk_id: str
    contract_title: str
    section_title: str
    raw_text: str
    char_start: int
    char_end: int


def _subdivide_large_section(
    section_title: str,
    raw_text: str,
    contract_title: str,
    base_id: str,
    char_offset: int,
) -> list[SectionChunk]:
    """Subdivide a section that exceeds MAX_CHUNK_CHARS into overlapping pieces."""
    sub_chunks: list[SectionChunk] = []
    start = 0
    part = 1
    text_len = len(raw_text)

    while start < text_len:
        end = min(start + TARGET_CHUNK_CHARS, text_len)
        if end < text_len:
            # Snap to paragraph or sentence boundary
            for sep in ("\n\n", ".\n", ". ", ";\n", "\n"):
                pos = raw_text.rfind(sep, start + CHUNK_OVERLAP_CHARS, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        slice_text = raw_text[start:end].strip()
        if slice_text:
            title = f"{section_title} (Part {part})" if part > 1 or end < text_len else section_title
            sub_chunks.append(
                SectionChunk(
                    chunk_id=f"{base_id}_p{part}",
                    contract_title=contract_title,
                    section_title=title,
                    raw_text=slice_text,
                    char_start=char_offset + start,
                    char_end=char_offset + end,
                )
            )
            part += 1

        if end >= text_len:
            break
        # ponytail: simple character overlap; tokenizers unneeded here
        start = max(start + 1, end - CHUNK_OVERLAP_CHARS)

    return sub_chunks


def chunk_contract_by_section(
    contract_text: str,
    contract_title: str = "untitled_contract",
) -> list[SectionChunk]:
    """
    Chunk contract text by legal headings (Articles, Sections, Exhibits).
    If no headings are found, partitions by paragraph sliding window.
    """
    if not contract_text or not contract_text.strip():
        return []

    matches = list(_HEADING_PATTERN.finditer(contract_text))

    # Fallback: if contract has fewer than 2 detected headings, treat as unstructured
    if len(matches) < 2:
        return _subdivide_large_section(
            section_title="General Contract Terms",
            raw_text=contract_text,
            contract_title=contract_title,
            base_id=f"{contract_title[:20]}_sec0",
            char_offset=0,
        )

    chunks: list[SectionChunk] = []
    
    # Text before the first heading (preamble, title, recitals)
    first_match = matches[0]
    if first_match.start() > 100:
        preamble = contract_text[:first_match.start()].strip()
        if preamble:
            chunks.extend(
                _subdivide_large_section(
                    section_title="Preamble & Recitals",
                    raw_text=preamble,
                    contract_title=contract_title,
                    base_id=f"{contract_title[:20]}_preamble",
                    char_offset=0,
                )
            )

    # Process each heading-delimited block
    for i, match in enumerate(matches):
        sec_start = match.start()
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(contract_text)
        
        # Heading line itself
        heading_line = match.group(0).strip().replace("\n", " ")
        raw_section = contract_text[sec_start:sec_end].strip()
        
        if not raw_section:
            continue

        base_id = f"{contract_title[:20]}_s{i+1}"
        if len(raw_section) > MAX_CHUNK_CHARS:
            chunks.extend(
                _subdivide_large_section(
                    section_title=heading_line,
                    raw_text=raw_section,
                    contract_title=contract_title,
                    base_id=base_id,
                    char_offset=sec_start,
                )
            )
        else:
            chunks.append(
                SectionChunk(
                    chunk_id=base_id,
                    contract_title=contract_title,
                    section_title=heading_line,
                    raw_text=raw_section,
                    char_start=sec_start,
                    char_end=sec_end,
                )
            )

    return chunks


if __name__ == "__main__":
    # Ponytail verification check
    sample_text = (
        "AGREEMENT between Party A and Party B.\n\n"
        "1. DEFINITIONS\nIn this agreement, terms mean what they say.\n\n"
        "2. TERMINATION\nEither party may terminate upon thirty (30) days notice.\n\n"
        "3. LIMITATION OF LIABILITY\nNeither party is liable for lost profits.\n"
    )
    result = chunk_contract_by_section(sample_text, contract_title="SampleAgreement")
    assert len(result) >= 3, f"Expected at least 3 sections, got {len(result)}"
    assert "TERMINATION" in result[1].section_title, f"Wrong title: {result[1].section_title}"
    assert "thirty (30) days" in result[1].raw_text, "Missing body text in section chunk"
    print(f"Self-check passed: {len(result)} section chunks created.")
    for c in result:
        print(f"  [{c.chunk_id}] {c.section_title}: {len(c.raw_text)} chars")
