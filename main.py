#!/usr/bin/env python3
"""
main.py
=======
Single entry point for the Contract Analysis Pipeline.

Usage:
    python main.py                      # run full pipeline (50 contracts)
    python main.py run --max 5         # quick test: only 5 contracts
    python main.py ask "indemnification cap"
    python main.py run --skip-download # use already-downloaded PDFs
"""
from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.logger import get_logger
from src.pipeline import run_pipeline
from src.output_writer import merge_batches

log = get_logger(__name__)


def cmd_run(args):
    """Run the extraction + summarization pipeline (optionally batched)."""
    batch_tag = None
    if args.batch:
        if args.batch.isdigit():
            batch_tag = f"batch_{int(args.batch):02d}"
        else:
            batch_tag = args.batch

    results = run_pipeline(
        max_contracts=args.max,
        skip_download=args.skip_download,
        offset=args.offset,
        batch_tag=batch_tag,
    )

    # Print a compact summary table for the console
    print("\n" + "=" * 70)
    print(f"{'CONTRACT':<35} {'PROVIDER':<14} {'STATUS'}")
    print("-" * 70)
    for r in results:
        status = "✓ ok" if not r.errors else f"⚠ {len(r.errors)} err(s)"
        print(f"{r.contract_id[:33]:<35} {r.provider:<14} {status}")
    print("=" * 70)
    print(f"Done. {len(results)} contracts analyzed. See /output for CSV & JSON.")



def cmd_merge(args):
    """Merge all batch_*.json files into final consolidated CSV + JSON."""
    csv_path, json_path = merge_batches()
    print(f"\nMerged all batches into:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")


def cmd_ask(args):
    """Run an end-to-end RAG query over the contract portfolio."""
    from src.rag.engine import ContractRAGEngine
    engine = ContractRAGEngine()

    if engine.vector_store.count() == 0:
        print("\n[!] ChromaDB is empty. Indexing first batch of contracts...")
        from src.data_loader import select_contracts
        records = select_contracts(count=args.max_contracts)
        for r in records:
            engine.index_contract(r.title, r.text)
        print(f"Indexed {engine.vector_store.count()} chunks into ChromaDB.")

    resp = engine.ask(args.query, top_k=args.top, filter_contract=args.contract)
    print("\n" + "=" * 70)
    print(f"QUERY: {resp.query}")
    print("=" * 70)
    print(f"\n{resp.answer}\n")
    print("-" * 70)
    print(f"CITATIONS ({len(resp.citations)} sources | Provider: {resp.provider}/{resp.model}):")
    print("-" * 70)
    for i, c in enumerate(resp.citations, 1):
        print(f"[{i}] {c.contract_title} — {c.section_title} (Match: {c.score:.1%})")
        snippet = c.raw_excerpt.strip()[:220].replace("\n", " ")
        print(f"    Excerpt: \"{snippet}...\"\n")


def cmd_index_rag(args):
    """Index contracts into ChromaDB for RAG."""
    from src.data_loader import select_contracts
    from src.rag.engine import ContractRAGEngine
    engine = ContractRAGEngine()
    records = select_contracts(count=args.max, offset=args.offset)
    print(f"\nIndexing {len(records)} contracts into ChromaDB...")
    total_added = 0
    for r in records:
        added = engine.index_contract(r.title, r.text)
        total_added += added
        print(f"  ✓ {r.title[:40]}... ({added} sections)")
    print(f"\nDone! Total indexed chunks in ChromaDB: {engine.vector_store.count()}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered contract analysis pipeline (CUAD dataset)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the pipeline (optionally in batches).")
    p_run.add_argument("--max", type=int, default=2,
                       help="Number of contracts to process in this run (default: 2).")
    p_run.add_argument("--offset", type=int, default=0,
                       help="Skip the first N contracts (for batched runs).")
    p_run.add_argument("--batch", type=str, default=None,
                       help="Batch tag (e.g. '01', '02'). Saves to output/batch_NN.csv/json.")
    p_run.add_argument("--skip-download", action="store_true", help="Skip dataset download.")
    p_run.set_defaults(func=cmd_run)

    p_merge = sub.add_parser("merge", help="Merge all batch files into final CSV + JSON.")
    p_merge.set_defaults(func=cmd_merge)

    p_ask = sub.add_parser("ask", help="RAG question-answering over contracts with citations.")
    p_ask.add_argument("query", help="Legal or operational question.")
    p_ask.add_argument("--top", type=int, default=3, help="Number of chunks to retrieve (default: 3).")
    p_ask.add_argument("--contract", type=str, default=None, help="Filter to a specific contract title.")
    p_ask.add_argument("--max-contracts", type=int, default=5, help="Number of contracts to index if empty (default: 5).")
    p_ask.set_defaults(func=cmd_ask)

    p_idx = sub.add_parser("index-rag", help="Index contracts into ChromaDB for RAG.")
    p_idx.add_argument("--max", type=int, default=5, help="Number of contracts to index (default: 5).")
    p_idx.add_argument("--offset", type=int, default=0, help="Offset into contract list (default: 0).")
    p_idx.set_defaults(func=cmd_index_rag)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
