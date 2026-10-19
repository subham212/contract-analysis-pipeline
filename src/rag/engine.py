"""
engine.py
=========
The RAG execution engine for legal contracts:
  1. Searches ChromaDB for relevant section summaries (Way B).
  2. Extracts the verbatim raw contract text and metadata.
  3. Injects the raw excerpts into a grounded anti-hallucination prompt.
  4. Generates an answer with verifiable legal citations via LLM router.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.logger import get_logger
from src.llm_client import get_router
from src.rag.section_chunker import chunk_contract_by_section, SectionChunk
from src.rag.summarizer import summarize_chunks_batch
from src.rag.vector_store import ContractVectorStore

log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert legal contract analyst.
Answer the user's question using ONLY the provided contract excerpts.
Rules:
- Be factual and precise. Quote relevant numbers, days, and monetary figures exactly.
- If the answer is not contained in the provided excerpts, state: "The provided contract documents do not specify this information."
- Always cite the contract name and section title for each fact you state.
- Do not invent terms or speculate.
"""

_USER_PROMPT_TEMPLATE = """\
CONTEXT FROM CONTRACT DOCUMENTS:
{context_blocks}

USER QUESTION:
{query}

ANSWER WITH CITATIONS:"""


@dataclass
class RAGCitation:
    """A single source excerpt supporting the answer."""
    contract_title: str
    section_title: str
    score: float
    raw_excerpt: str
    summary: str


@dataclass
class RAGResponse:
    """Complete RAG response containing the synthesized answer and citations."""
    query: str
    answer: str
    citations: list[RAGCitation] = field(default_factory=list)
    provider: str = ""
    model: str = ""


class ContractRAGEngine:
    """Orchestrates ingestion, retrieval, and generation."""

    def __init__(
        self,
        vector_store: Optional[ContractVectorStore] = None,
        router=None,
    ):
        self.vector_store = vector_store or ContractVectorStore()
        self.router = router or get_router()

    def index_contract(self, contract_title: str, contract_text: str) -> int:
        """Chunk, summarize, and index a single contract into ChromaDB."""
        chunks = chunk_contract_by_section(contract_text, contract_title)
        if not chunks:
            return 0
        summaries = summarize_chunks_batch(chunks, router=self.router)
        return self.vector_store.add_chunks(chunks, summaries)

    def ask(
        self,
        query: str,
        top_k: int = 3,
        filter_contract: Optional[str] = None,
    ) -> RAGResponse:
        """
        Ask a question across indexed contracts.
        Retrieves top-k chunks, assembles prompt, and generates grounded answer.
        """
        query_clean = query.strip()
        if not query_clean:
            return RAGResponse(query=query, answer="Please provide a valid question.")

        # 1. Retrieve top-k matching documents from ChromaDB
        search_hits = self.vector_store.search(
            query=query_clean,
            top_k=top_k,
            filter_contract=filter_contract,
        )

        if not search_hits:
            return RAGResponse(
                query=query,
                answer="No relevant contract sections found in the vector index.",
            )

        # 2. Build citations and context blocks using Way B raw_content
        context_blocks: list[str] = []
        citations: list[RAGCitation] = []

        for i, (doc, dist) in enumerate(search_hits, 1):
            meta = doc.metadata or {}
            contract_title = meta.get("contract_title", "Unknown Contract")
            section_title = meta.get("section_title", "General Section")
            raw_content = meta.get("raw_content") or doc.page_content

            # ponytail: convert distance to normalized similarity percentage (0..1)
            score = max(0.0, min(1.0, 1.0 - (dist / 2.0 if dist > 1.0 else dist)))

            citations.append(
                RAGCitation(
                    contract_title=contract_title,
                    section_title=section_title,
                    score=round(score, 3),
                    raw_excerpt=raw_content,
                    summary=doc.page_content,
                )
            )

            context_blocks.append(
                f"[Source {i}] Contract: {contract_title}\n"
                f"Section: {section_title}\n"
                f"Excerpt:\n{raw_content}\n"
            )

        # 3. Assemble prompt and generate answer
        full_context = "\n---\n".join(context_blocks)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            context_blocks=full_context,
            query=query_clean,
        )

        llm_out = self.router.generate(user_prompt, system=_SYSTEM_PROMPT)

        return RAGResponse(
            query=query_clean,
            answer=llm_out.text.strip(),
            citations=citations,
            provider=llm_out.provider,
            model=llm_out.model,
        )


if __name__ == "__main__":
    # Ponytail verification self-check with dummy router and temporary vector store
    import tempfile
    from langchain_core.embeddings import Embeddings
    from src.rag.vector_store import EMBED_DIM

    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.1] * EMBED_DIM for _ in texts]
        def embed_query(self, text):
            return [0.1] * EMBED_DIM

    class MockRouter:
        def generate(self, prompt: str, system: Optional[str] = None):
            from types import SimpleNamespace
            assert "CONTEXT FROM CONTRACT DOCUMENTS" in prompt
            assert "Acme SLA" in prompt
            return SimpleNamespace(
                text="Under Section 8 of the Acme SLA, either party may terminate upon 30 days notice.",
                provider="mock-llm",
                model="mock-model",
            )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        store = ContractVectorStore(persist_dir=Path(td), embeddings=MockEmbeddings())
        engine = ContractRAGEngine(vector_store=store, router=MockRouter())

        # Index 1 contract
        sample_contract = (
            "1. DEFINITIONS\nAll terms defined herein.\n\n"
            "8. TERMINATION CLAUSE\n"
            "Either party may terminate this agreement without cause upon thirty (30) days prior written notice."
        )
        indexed_count = engine.index_contract("Acme SLA", sample_contract)
        assert indexed_count >= 1, "Failed to index sample contract"

        # Ask a question
        response = engine.ask("What is the notice period to terminate without cause?")
        assert len(response.citations) >= 1
        assert "30 days notice" in response.answer
        assert any(c.contract_title == "Acme SLA" for c in response.citations)
        assert any("TERMINATION" in c.section_title for c in response.citations)
        print("Self-check passed: ContractRAGEngine end-to-end verified successfully!")
        print(f"  Q: {response.query}")
        print(f"  A: {response.answer}")
        print(f"  Citation: {response.citations[0].contract_title} - {response.citations[0].section_title}")
