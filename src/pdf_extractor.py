"""
pdf_extractor.py
================
Extracts raw text from contract PDFs and normalizes it.

Why pdfplumber?
  - Handles multi-column layouts common in legal docs.
  - More accurate than PyPDF2 for text reconstruction.
  - Pure-Python, no system dependencies.

Normalization steps:
  1. Collapse runs of whitespace.
  2. Fix hyphenated line-breaks ("lia-bility" → "liability").
  3. Strip page numbers / header artifacts.
  4. Normalize unicode quotes/dashes to ASCII.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pdfplumber

from .logger import get_logger

log = get_logger(__name__)

# Regex patterns compiled once for performance
_RE_WHITESPACE = re.compile(r"[ \t]+")
_RE_NEWLINES = re.compile(r"\n{3,}")
_RE_HYPHENATED = re.compile(r"(\w)-\n(\w)")
_RE_PAGE_NUM = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file, page by page.

    Returns the concatenated raw text (before normalization).
    Raises RuntimeError on unreadable/corrupt PDFs.
    """
    pdf_path = Path(pdf_path)
    pages_text: list[str] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF {pdf_path.name}: {exc}") from exc

    return "\n\n".join(pages_text)


def normalize_text(text: str) -> str:
    """
    Clean and normalize extracted text.

    Steps:
      - Unicode → ASCII (smart quotes, em-dashes).
      - Join hyphenated line-breaks.
      - Drop standalone page-number lines.
      - Collapse excessive whitespace/newlines.
    """
    if not text:
        return ""

    # 1. Unicode normalization (keeps the text readable & model-friendly)
    text = unicodedata.normalize("NFKC", text)

    # 2. Join words split across lines: "liab-\nility" → "liability"
    text = _RE_HYPHENATED.sub(r"\1\2", text)

    # 3. Remove standalone page-number lines
    cleaned_lines = []
    for line in text.splitlines():
        if _RE_PAGE_NUM.match(line.strip()):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 4. Collapse runs of spaces/tabs into a single space
    text = _RE_WHITESPACE.sub(" ", text)

    # 5. Collapse 3+ newlines into exactly 2
    text = _RE_NEWLINES.sub("\n\n", text)

    return text.strip()


def extract_and_clean(pdf_path: Path) -> str:
    """Convenience: extract + normalize in one call."""
    raw = extract_text_from_pdf(pdf_path)
    return normalize_text(raw)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pdf_extractor <path/to/contract.pdf>")
        sys.exit(1)

    cleaned = extract_and_clean(Path(sys.argv[1]))
    print(f"Extracted {len(cleaned)} chars.")
    print("---")
    print(cleaned[:1000])
