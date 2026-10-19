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


def test_section_chunker():
    from src.rag.section_chunker import chunk_contract_by_section
    raw = (
        "PREAMBLE\nThis agreement is made...\n\n"
        "SECTION 1. DEFINITIONS\nTerms defined here.\n\n"
        "SECTION 2. TERMINATION\nNotice period is 30 days.\n"
    )
    chunks = chunk_contract_by_section(raw, "TestContract")
    assert len(chunks) >= 2
    assert any("TERMINATION" in c.section_title for c in chunks)
    assert any("DEFINITIONS" in c.section_title for c in chunks)


def test_summarizer_cache():
    import tempfile
    from src.rag.section_chunker import SectionChunk
    from src.rag.summarizer import SummaryCache, summarize_chunk
    
    class FakeRouter:
        def generate(self, p, system=None):
            from types import SimpleNamespace
            return SimpleNamespace(text="Fake summary of section.", provider="mock")

    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "cache.json"
        cache = SummaryCache(cp)
        chunk = SectionChunk("c1", "T", "Sec 1", "A" * 300, 0, 300)
        s1 = summarize_chunk(chunk, router=FakeRouter(), cache=cache)
        assert s1 == "Fake summary of section."
        assert cache.get("c1") == s1
        # Second call does not need router
        s2 = summarize_chunk(chunk, router=None, cache=cache)
        assert s2 == s1


def test_chromadb_way_b():
    import tempfile
    from langchain_core.embeddings import Embeddings
    from src.rag.section_chunker import SectionChunk
    from src.rag.vector_store import ContractVectorStore, EMBED_DIM

    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.05] * EMBED_DIM for _ in texts]
        def embed_query(self, text):
            return [0.05] * EMBED_DIM

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        store = ContractVectorStore(persist_dir=Path(td), embeddings=MockEmbeddings())
        chunk = SectionChunk("c1", "TestContract", "Sec 1", "Raw contract clause text here.", 0, 30)
        store.add_chunks([chunk], {"c1": "Summary of clause."})
        assert store.count() == 1
        
        results = store.search("clause", top_k=1)
        assert len(results) == 1
        doc, _ = results[0]
        # Way B verification:
        assert doc.page_content == "Summary of clause."
        assert doc.metadata["raw_content"] == "Raw contract clause text here."
        assert doc.metadata["section_title"] == "Sec 1"


def test_rag_engine_end_to_end():
    import tempfile
    from langchain_core.embeddings import Embeddings
    from src.rag.vector_store import ContractVectorStore, EMBED_DIM
    from src.rag.engine import ContractRAGEngine

    class FastEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.05] * EMBED_DIM for _ in texts]
        def embed_query(self, text):
            return [0.05] * EMBED_DIM

    class MockLLM:
        def generate(self, prompt, system=None):
            from types import SimpleNamespace
            return SimpleNamespace(
                text="Party A has 30 days to terminate under Section 2.",
                provider="mock-prov",
                model="mock-model",
            )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        store = ContractVectorStore(persist_dir=Path(td), embeddings=FastEmbeddings())
        engine = ContractRAGEngine(vector_store=store, router=MockLLM())
        contract_text = "SECTION 1. DEFINITIONS\nTerms.\n\nSECTION 2. TERMINATION\nNotice is 30 days."
        engine.index_contract("TestCo Agreement", contract_text)
        
        resp = engine.ask("How many days to terminate?")
        assert "30 days" in resp.answer
        assert len(resp.citations) >= 1
        assert any(c.contract_title == "TestCo Agreement" for c in resp.citations)


def test_app_rag_endpoint():
    from fastapi.testclient import TestClient
    import app
    client = TestClient(app.app)
    # Test HTML UI contains RAG view
    r = client.get("/")
    assert r.status_code == 200
    assert "view-rag" in r.text
    assert "AI Legal Assistant" in r.text
    # Ensure obsolete semantic search endpoint returns 404
    r_search = client.get("/api/search?q=test")
    assert r_search.status_code == 404


def test_contract_embeddings():
    from src.rag.vector_store import ContractEmbeddings, EMBED_DIM
    emb = ContractEmbeddings()
    vecs = emb.embed_documents(["Test embedding clause"])
    assert len(vecs) == 1
    assert len(vecs[0]) == EMBED_DIM


def run_all():
    """Run all tests without pytest."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
