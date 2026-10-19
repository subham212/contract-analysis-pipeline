"""
vector_store.py
===============
ChromaDB persistent vector store implementing the Way B pattern:
  - page_content = 2-3 sentence search summary (vectorized for search)
  - metadata["raw_content"] = full original contract text (retrieved for LLM)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.logger import get_logger
from src.rag.section_chunker import SectionChunk

log = get_logger(__name__)

DEFAULT_PERSIST_DIR = Path(settings.output_dir) / "chroma_db"
COLLECTION_NAME = "contract_sections"
EMBED_DIM = 768


class ContractEmbeddings(Embeddings):
    """Embeddings wrapper: uses ZenMux/OpenAI API with deterministic hash fallback."""

    def __init__(self):
        self._openai_client = None

        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    base_url=settings.openai_base_url or None,
                    api_key=settings.openai_api_key,
                )
                log.info("Embeddings: initialized OpenAI/ZenMux (%s, dim=%d)",
                         settings.openai_embedding_model, EMBED_DIM)
            except Exception as exc:
                log.warning("OpenAI/ZenMux client init failed for embeddings: %s", exc)

    def _hash_embed(self, text: str) -> list[float]:
        # ponytail: deterministic fallback if API is unavailable
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for tok in text.lower().split():
            vec[hash(tok) % EMBED_DIM] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # 1. Primary: ZenMux / OpenAI embeddings
        if self._openai_client:
            try:
                model_name = settings.openai_embedding_model or "openai/text-embedding-3-small"
                all_embeddings: list[list[float]] = []
                batch_size = 64
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    resp = self._openai_client.embeddings.create(
                        input=batch,
                        model=model_name,
                        dimensions=EMBED_DIM,
                    )
                    all_embeddings.extend([item.embedding for item in resp.data])
                return all_embeddings
            except Exception as exc:
                log.warning("ZenMux embedding failed, falling back to hash: %s", exc)

        # 2. Fallback: Local deterministic hash
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ContractVectorStore:
    """Manages persistent ChromaDB collection with Way B document storage."""

    def __init__(self, persist_dir: Optional[Path] = None, embeddings: Optional[Embeddings] = None):
        self.persist_dir = Path(persist_dir or DEFAULT_PERSIST_DIR)
        self.embeddings = embeddings or ContractEmbeddings()
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def add_chunks(self, chunks: list[SectionChunk], summaries: dict[str, str]) -> int:
        """
        Store chunks using Way B:
        Summary is embedded; raw text is preserved in metadata.
        Idempotent by chunk_id.
        """
        if not chunks:
            return 0

        docs: list[Document] = []
        ids: list[str] = []

        for chunk in chunks:
            summary = summaries.get(chunk.chunk_id) or chunk.raw_text[:250]
            doc = Document(
                page_content=summary,
                metadata={
                    "raw_content": chunk.raw_text,
                    "contract_title": chunk.contract_title,
                    "section_title": chunk.section_title,
                    "chunk_id": chunk.chunk_id,
                },
            )
            docs.append(doc)
            ids.append(chunk.chunk_id)

        # ponytail: Chroma upsert via IDs prevents duplicate entries on re-run
        self.vector_store.add_documents(documents=docs, ids=ids)
        log.info("Indexed %d chunks in ChromaDB at %s", len(docs), self.persist_dir)
        return len(docs)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_contract: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search summaries by query; returns top-k matching (Document, similarity_score).
        The Document metadata contains the raw_content for generation.
        """
        filter_dict = {"contract_title": filter_contract} if filter_contract else None
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict,
        )
        return results

    def count(self) -> int:
        return self.vector_store._collection.count()


if __name__ == "__main__":
    # Ponytail verification self-check with temporary storage
    import tempfile

    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.1] * EMBED_DIM for _ in texts]
        def embed_query(self, text):
            return [0.1] * EMBED_DIM

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        store = ContractVectorStore(persist_dir=Path(td), embeddings=DummyEmbeddings())
        
        chunk = SectionChunk(
            chunk_id="chk_01",
            contract_title="Acme_SLA",
            section_title="8. TERMINATION",
            raw_text="Either party may terminate upon 30 days notice.",
            char_start=0,
            char_end=48,
        )
        summary = "Section 8 allows termination upon 30 days notice."
        
        # Test add
        store.add_chunks([chunk], {chunk.chunk_id: summary})
        assert store.count() == 1, f"Expected 1 document, found {store.count()}"

        # Test search & Way B raw retrieval
        hits = store.search("terminate notice", top_k=1)
        assert len(hits) == 1
        matched_doc, score = hits[0]
        assert matched_doc.page_content == summary, "Summary mismatch in page_content"
        assert matched_doc.metadata["raw_content"] == chunk.raw_text, "Raw content missing from metadata"
        assert matched_doc.metadata["section_title"] == "8. TERMINATION"
        print(f"Self-check passed: ChromaDB Way B verified! (Count={store.count()})")
