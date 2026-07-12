"""
output_writer.py
================
Persist pipeline results to disk in CSV and JSON formats.

CSV is for HR/analysts who want a spreadsheet view.
JSON is for downstream programmatic use (preserves structure).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .logger import get_logger

log = get_logger(__name__)


@dataclass
class ContractResult:
    """The complete analysis result for one contract."""
    contract_id: str
    filename: str
    summary: str = ""
    termination_clause: str = "Not found"
    confidentiality_clause: str = "Not found"
    liability_clause: str = "Not found"
    num_chunks: int = 0
    char_count: int = 0
    provider: str = ""
    errors: list[str] = field(default_factory=list)


def write_results(results: list[ContractResult], output_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Write results to both CSV and JSON.

    Returns: (csv_path, json_path)
    """
    output_dir = output_dir or settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ─── CSV (flat, analyst-friendly) ────────────────────────────
    csv_path = output_dir / f"contracts_analysis_{timestamp}.csv"
    columns = [
        "contract_id", "filename", "summary",
        "termination_clause", "confidentiality_clause", "liability_clause",
        "num_chunks", "char_count", "provider",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # CSV cannot hold lists — collapse errors into a string
            row["errors"] = "; ".join(row.get("errors", [])) if row.get("errors") else ""
            writer.writerow(row)

    # ─── JSON (structured, preserves everything) ────────────────
    json_path = output_dir / f"contracts_analysis_{timestamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "primary": f"Gemini/{settings.gemini_model}" if settings.gemini_api_key else None,
            "fallback": (
                f"OpenAI-compat/{settings.openai_model}"
                if settings.openai_api_key else None
            ),
        },
        "extraction_mode": settings.extraction_mode,
        "total_contracts": len(results),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Wrote CSV : %s", csv_path)
    log.info("Wrote JSON: %s", json_path)
    return csv_path, json_path


def write_latest_symlink(results: list[ContractResult], output_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Also write stable 'latest' files so HR can find results without timestamps.
    (On Windows these are real files, not symlinks, for portability.)
    """
    csv_path, json_path = write_results(results, output_dir)
    latest_csv = csv_path.parent / "latest.csv"
    latest_json = json_path.parent / "latest.json"
    latest_csv.write_bytes(csv_path.read_bytes())
    latest_json.write_bytes(json_path.read_bytes())
    return latest_csv, latest_json
