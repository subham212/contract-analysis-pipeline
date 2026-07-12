#!/usr/bin/env python3
"""
main.py
=======
Single entry point for the Contract Analysis Pipeline.

Usage:
    python main.py                      # run full pipeline (50 contracts)
    python main.py --max 5             # quick test: only 5 contracts
    python main.py --search "indemnification cap"
    python main.py --skip-download     # use already-downloaded PDFs
"""
from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.logger import get_logger
from src.pipeline import run_pipeline

log = get_logger(__name__)


def cmd_run(args):
    """Run the full extraction + summarization pipeline."""
    results = run_pipeline(max_contracts=args.max, skip_download=args.skip_download)

    # Print a compact summary table for the console
    print("\n" + "=" * 70)
    print(f"{'CONTRACT':<35} {'PROVIDER':<14} {'STATUS'}")
    print("-" * 70)
    for r in results:
        status = "✓ ok" if not r.errors else f"⚠ {len(r.errors)} err(s)"
        print(f"{r.contract_id[:33]:<35} {r.provider:<14} {status}")
    print("=" * 70)
    print(f"Done. {len(results)} contracts analyzed. See /output for CSV & JSON.")


def cmd_search(args):
    """Run a semantic search over previously-extracted clauses."""
    import json
    from pathlib import Path

    from src.semantic_search import SemanticSearcher

    index_path = Path(settings.output_dir) / "semantic_index.json"
    if not index_path.exists():
        print("No semantic index found. Run the pipeline first: python main.py")
        sys.exit(1)

    raw = json.loads(index_path.read_text(encoding="utf-8"))
    searcher = SemanticSearcher()
    searcher.clauses = raw["clauses"]
    if raw.get("embeddings"):
        import numpy as np
        searcher.embeddings = np.array(raw["embeddings"], dtype=np.float32)

    hits = searcher.query(args.query, top_k=args.top)
    if not hits:
        print("No results.")
        return

    print(f"\nTop {len(hits)} matches for: {args.query!r}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h.clause_type}] {h.filename}  (score: {h.score:.3f})")
        print(f"   {h.text[:200]}{'...' if len(h.text) > 200 else ''}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered contract analysis pipeline (CUAD dataset)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline.")
    p_run.add_argument("--max", type=int, default=None, help="Max contracts to process.")
    p_run.add_argument("--skip-download", action="store_true", help="Skip dataset download.")
    p_run.set_defaults(func=cmd_run)

    p_search = sub.add_parser("search", help="Semantic search over extracted clauses.")
    p_search.add_argument("query", help="Natural-language query.")
    p_search.add_argument("--top", type=int, default=5, help="Number of results.")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
