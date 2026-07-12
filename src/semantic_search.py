"""
semantic_search.py  (BONUS)
===========================
Search extracted clauses by MEANING, not just keywords.

How it works:
  1. Convert every clause to an embedding (a vector of numbers).
  2. When the user types a query, embed it the same way.
  3. Rank clauses by cosine similarity (how close the vectors are in meaning).

We use Gemini's embedding model (text-embedding-004) as the embedder because
it is free and high quality. Falls back gracefully if unavailable.

This is a classic RAG (Retrieval-Augmented Generation) building block.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import settings
from .logger import get_logger

log = get_logger(__name__)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # output dim of gemini-embedding-001


@dataclass
class SearchHit:
    """A single search result."""
    contract_id: str
    filename: str
    clause_type: str
    text: str
    score: float  # cosine similarity, 0..1


class SemanticSearcher:
    """Build an embedding index over clauses and answer queries."""

    def __init__(self):
        self.clauses: list[dict] = []        # [{contract_id, filename, clause_type, text}]
        self.embeddings: Optional[np.ndarray] = None  # shape (N, D)
        self._embed_fn = None

    def _resolve_embed_fn(self):
        """Pick an embedding source: Gemini first, else heuristic."""
        if self._embed_fn is not None:
            return self._embed_fn

        if settings.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=settings.gemini_api_key)

                def gemini_embed(texts: list[str]) -> np.ndarray:
                    # New SDK embed_content signature
                    resp = client.models.embed_content(
                        model=GEMINI_EMBEDDING_MODEL,
                        contents=texts,
                        config={"task_type": "RETRIEVAL_DOCUMENT",
                                "output_dimensionality": EMBED_DIM},
                    )
                    # resp.embeddings is a list of Embedding objects with .values
                    vecs = [e.values for e in resp.embeddings]
                    return np.array(vecs, dtype=np.float32)

                self._embed_fn = gemini_embed
                log.info("Embedding backend: Gemini (%s)", GEMINI_EMBEDDING_MODEL)
                return self._embed_fn
            except Exception as exc:
                log.warning("Gemini embedding unavailable, using fallback: %s", exc)

        # Deterministic fallback: simple hashed-bag (no extra deps).
        # Quality is lower than real embeddings but keeps the feature working.
        def hash_embed(texts: list[str]) -> np.ndarray:
            out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                for tok in t.lower().split():
                    h = hash(tok) % EMBED_DIM
                    out[i, h] += 1.0
            # L2 normalize rows
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return out / norms

        self._embed_fn = hash_embed
        log.info("Embedding backend: hashed-bag fallback")
        return self._embed_fn

    def build_index(self, results: list[dict]) -> None:
        """
        Build the index from pipeline results (list of ContractResult dicts).

        Each result contributes up to 3 clauses (termination, confidentiality, liability).
        """
        self.clauses = []
        clause_fields = [
            ("termination_clause", "termination"),
            ("confidentiality_clause", "confidentiality"),
            ("liability_clause", "liability"),
        ]
        for r in results:
            for field_name, clause_type in clause_fields:
                text = (r.get(field_name) or "").strip()
                if text and text.lower() != "not found":
                    self.clauses.append({
                        "contract_id": r["contract_id"],
                        "filename": r["filename"],
                        "clause_type": clause_type,
                        "text": text,
                    })

        if not self.clauses:
            log.warning("No clauses to index for semantic search.")
            self.embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
            return

        embed_fn = self._resolve_embed_fn()
        texts = [c["text"] for c in self.clauses]

        # Embed in small batches to respect API limits
        batch = 16
        all_vecs = []
        for i in range(0, len(texts), batch):
            chunk_texts = texts[i:i + batch]
            try:
                vecs = embed_fn(chunk_texts)
                all_vecs.append(vecs)
            except Exception as exc:
                log.warning("Embedding batch %d failed: %s", i, exc)
                # fall back to hash embed for this batch
                all_vecs.append(_hash_embed_fallback(chunk_texts))

        self.embeddings = np.vstack(all_vecs)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        log.info("Indexed %d clauses for semantic search.", len(self.clauses))

    def query(self, question: str, top_k: int = 5) -> list[SearchHit]:
        """Return the top-k clauses most similar to the question."""
        if self.embeddings is None or len(self.clauses) == 0:
            return []

        embed_fn = self._resolve_embed_fn()
        try:
            q_vec = embed_fn([question])[0]
        except Exception as exc:
            log.warning("Query embedding failed: %s", exc)
            from .semantic_search import _hash_embed_fallback
            q_vec = _hash_embed_fallback([question])[0]

        # Normalize query vector
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Cosine similarity (vectors are already normalized)
        scores = self.embeddings @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            SearchHit(
                contract_id=self.clauses[i]["contract_id"],
                filename=self.clauses[i]["filename"],
                clause_type=self.clauses[i]["clause_type"],
                text=self.clauses[i]["text"],
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    def save_index(self, path: Path) -> None:
        """Persist the index to disk (for reuse / inspection)."""
        payload = {
            "clauses": self.clauses,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        log.info("Saved semantic index (%d clauses) to %s", len(self.clauses), path)


def _hash_embed_fallback(texts: list[str]) -> np.ndarray:
    """Module-level fallback so query() can recover from embed errors."""
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = hash(tok) % EMBED_DIM
            out[i, h] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms
