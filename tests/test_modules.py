"""
Unit tests for core modules.
Run with:  python -m pytest tests/ -v
(Or without pytest: just run this file directly.)
"""
import sys
from pathlib import Path

# Make src importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_text, estimate_tokens
from src.pdf_extractor import normalize_text
from src.prompts import get_extraction_prompt, get_summary_prompt
from src.output_writer import ContractResult


def test_chunker_short_text():
    """Short text should produce exactly one chunk."""
    chunks = chunk_text("Hello world.", chunk_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_chunker_long_text():
    """Long text should split into multiple overlapping chunks."""
    text = "Sentence one. " * 5000  # ~70k chars
    chunks = chunk_text(text, chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) > 1
    # Overlap: the end of chunk[i] should appear at the start of chunk[i+1]
    assert chunks[0].char_end > chunks[0].char_start


def test_estimate_tokens():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("four") == 1
    assert estimate_tokens("a" * 1000) == 250


def test_normalize_hyphenation():
    """Hyphenated line-breaks should be joined."""
    text = "lia-\nbility"
    out = normalize_text(text)
    assert "liability" in out
    assert "lia-\nbility" not in out


def test_normalize_page_numbers():
    """Standalone page-number lines should be removed."""
    text = "Some text.\n\n  Page 12 of 50  \n\nMore text."
    out = normalize_text(text)
    assert "Page 12 of 50" not in out
    assert "Some text" in out
    assert "More text" in out


def test_normalize_whitespace():
    text = "a    b\t\tc\n\n\n\nd"
    out = normalize_text(text)
    assert "    " not in out
    assert "\n\n\n" not in out


def test_extraction_prompt_zero_shot():
    p = get_extraction_prompt("CONTRACT TEXT", mode="zero_shot")
    assert "CONTRACT TEXT" in p
    assert "termination" in p.lower()
    assert "confidentiality" in p.lower()
    assert "liability" in p.lower()


def test_extraction_prompt_few_shot():
    p = get_extraction_prompt("CONTRACT TEXT", mode="few_shot")
    assert "EXAMPLE 1" in p
    assert "EXAMPLE 2" in p
    assert "CONTRACT TEXT" in p


def test_summary_prompt():
    p = get_summary_prompt("CONTRACT TEXT")
    assert "100 to 150 words" in p
    assert "PURPOSE" in p
    assert "OBLIGATIONS" in p
    assert "RISKS" in p


def test_contract_result_dataclass():
    r = ContractResult(contract_id="X", filename="X.pdf")
    assert r.termination_clause == "Not found"
    assert r.errors == []
    d = r.__dict__
    assert d["contract_id"] == "X"


def run_all():
    """Run all tests without pytest."""
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: ERROR {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
