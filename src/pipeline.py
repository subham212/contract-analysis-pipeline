"""
pipeline.py
===========
The orchestrator. Ties together every module into a clean end-to-end flow.

Flow:
    load PDFs → extract text → clean → chunk →
    (per chunk) extract clauses → (merge) → summarize →
    collect results → write CSV/JSON → (optional) build semantic index
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .chunker import chunk_text
from .config import settings
from .data_loader import ContractRecord, download_cuad, load_contracts, select_contracts
from .llm_client import LLMError, get_router
from .logger import get_logger
from .output_writer import ContractResult, write_batch, write_latest_symlink
from .prompts import SYSTEM_PROMPT, get_extraction_prompt, get_summary_prompt

log = get_logger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────

def _safe_json_parse(raw: str) -> dict:
    """
    Parse JSON from an LLM response that may include code fences or prose.
    Tries to find the first {...} block and parse it.
    """
    if not raw:
        return {}
    # Strip markdown code fences if present
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    # Find the outermost JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _contract_id_from_filename(filename: str) -> str:
    """Stable contract id, e.g. 'LendingTreeLOA.pdf' → 'LendingTreeLOA'."""
    return Path(filename).stem


# ─── Per-contract processing ─────────────────────────────────────

def process_contract(record: ContractRecord, router) -> ContractResult:
    """
    Process a single contract end-to-end.

    For contracts that exceed the model's context window we:
      1. Split into overlapping chunks.
      2. Run clause extraction on each chunk.
      3. Merge clause results (first non-empty wins per clause type).
      4. Summarize using the full (or truncated) cleaned text.
    """
    contract_id = record.title
    # filename mirrors the CUAD title for traceability (no real PDF here)
    filename = f"{record.title}.txt"
    result = ContractResult(
        contract_id=contract_id,
        filename=filename,
    )

    # 1. Use the contract text directly (CUAD provides already-extracted text)
    clean_text = record.text
    result.char_count = len(clean_text)
    if not clean_text or len(clean_text) < 200:
        result.errors.append("Contract text too short or empty.")
        result.summary = "Unable to extract sufficient text from this contract."
        return result

    # 2. Chunk
    chunks = chunk_text(clean_text)
    result.num_chunks = len(chunks)
    log.debug("%s: %d chars → %d chunks", contract_id, len(clean_text), len(chunks))

    # 3. Extract clauses from each chunk
    merged = {
        "termination": "Not found",
        "confidentiality": "Not found",
        "liability": "Not found",
    }

    for chunk in chunks:
        prompt = get_extraction_prompt(chunk.text, mode=settings.extraction_mode)
        try:
            llm_out = router.generate(prompt, system=SYSTEM_PROMPT)
            result.provider = result.provider or llm_out.provider
            parsed = _safe_json_parse(llm_out.text)
        except LLMError as exc:
            result.errors.append(f"Extraction failed on chunk {chunk.index}: {exc}")
            continue

        # Merge: keep the first non-empty, non-"Not found" answer
        for key in merged:
            val = (parsed.get(key) or "").strip()
            if val and val.lower() != "not found" and merged[key].lower() == "not found":
                merged[key] = val

    result.termination_clause = merged["termination"]
    result.confidentiality_clause = merged["confidentiality"]
    result.liability_clause = merged["liability"]

    # 4. Summarize using the full text (or a generous truncation if huge)
    summary_text = clean_text
    # Cap to ~120k chars (~30k tokens) to be safe across providers
    if len(summary_text) > 120_000:
        summary_text = summary_text[:120_000]
        log.debug("%s: truncated text for summary to 120k chars", contract_id)

    try:
        summary_out = router.generate(
            get_summary_prompt(summary_text), system=SYSTEM_PROMPT
        )
        result.provider = result.provider or summary_out.provider
        result.summary = summary_out.text.strip()
    except LLMError as exc:
        result.errors.append(f"Summarization failed: {exc}")
        result.summary = "Summary generation failed."

    return result


# ─── Full pipeline run ───────────────────────────────────────────

def run_pipeline(
    max_contracts: Optional[int] = None,
    skip_download: bool = False,
    offset: int = 0,
    batch_tag: Optional[str] = None,
) -> list[ContractResult]:
    """
    Run the full pipeline end-to-end.

    Args:
        max_contracts: override settings.max_contracts.
        skip_download: if True, assume dataset is already present.
        offset: skip the first `offset` contracts (for batched processing).
        batch_tag: if set, save outputs under a batch-specific name
                   (e.g. batch_03) so runs don't overwrite each other.
    """
    log.info("=" * 64)
    log.info("Contract Analysis Pipeline — starting")
    if offset:
        log.info("Batch mode: offset=%d", offset)
    if batch_tag:
        log.info("Batch tag: %s", batch_tag)
    log.info("=" * 64)

    problems = settings.validate()
    if problems:
        for p in problems:
            log.error("CONFIG ERROR: %s", p)
        raise SystemExit("Configuration invalid. Fix the issues above and retry.")

    # 1. Get contracts
    if not skip_download:
        download_cuad()
    records = load_contracts()
    contracts = select_contracts(
        records, count=max_contracts or settings.max_contracts, offset=offset
    )
    log.info("Processing %d contracts...", len(contracts))

    # 2. Init LLM router (failover chain)
    router = get_router()

    # 3. Process each contract
    results: list[ContractResult] = []
    for record in tqdm(contracts, desc="Analyzing contracts", unit="doc"):
        try:
            res = process_contract(record, router)
        except Exception as exc:  # noqa: BLE001 - never crash the whole run
            log.error("Unexpected failure on %s: %s", record.title, exc)
            res = ContractResult(
                contract_id=record.title,
                filename=f"{record.title}.txt",
                errors=[f"Unexpected error: {exc}"],
                summary="Processing failed unexpectedly.",
            )
        results.append(res)

    # 4. Write outputs (batch-specific if tagged, else latest)
    if batch_tag:
        csv_path, json_path = write_batch(results, batch_tag)
    else:
        csv_path, json_path = write_latest_symlink(results)
    log.info("Pipeline complete. %d/%d contracts processed without error.",
             sum(1 for r in results if not r.errors), len(results))
    log.info("Results: %s | %s", csv_path, json_path)

    # 5. Bonus: semantic search index
    if settings.enable_semantic_search:
        try:
            from .semantic_search import SemanticSearcher
            searcher = SemanticSearcher()
            searcher.build_index([r.__dict__ for r in results])
            index_path = settings.output_dir / "semantic_index.json"
            searcher.save_index(index_path)
            log.info("Semantic search index built (%d clauses).", len(searcher.clauses))
        except Exception as exc:  # noqa: BLE001
            log.warning("Semantic search skipped: %s", exc)

    return results
