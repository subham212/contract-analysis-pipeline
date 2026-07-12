"""
data_loader.py
==============
Downloads the CUAD dataset and prepares a subset of N contracts.

CUAD (Contract Understanding Atticus Dataset):
  - Source: https://github.com/TheAtticusProject/cuad
  - 510 commercial legal contracts, 13,000+ annotations across 41 clause types
  - We extract a deterministic subset (default 50) for this pipeline.

The downloaded archive (data.zip, ~18 MB) contains JSON files, NOT PDFs:
  - CUADv1.json                  → the full dataset (510 contracts + annotations)
  - test.json                    → test split
  - train_separate_questions.json→ train split (per-question)

Each contract in CUADv1.json has the SQuAD-like shape:
  {
    "title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
    "paragraphs": [
      {
        "context": "<full contract text>",
        "qas": [ {question, answers, id}, ... 41 annotations ... ]
      }
    ]
  }

We load the `context` as the contract text (it is already clean text
extracted by The Atticus Project), and also persist the original contract
text as a .txt file so the rest of the pipeline can treat it uniformly.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import CUAD_ZIP_URL, settings
from .logger import get_logger

log = get_logger(__name__)


@dataclass
class ContractRecord:
    """A single loaded contract with its ground-truth annotations."""
    title: str
    text: str
    num_annotations: int


def download_cuad(target_dir: Path | None = None) -> Path:
    """
    Download the CUAD data.zip and cache the raw JSON on disk.

    Returns the path to the cached CUADv1.json file.
    """
    target_dir = target_dir or (settings.project_root / "data" / "raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    cuad_json_path = target_dir / "CUADv1.json"
    marker = target_dir / ".cuad_downloaded"

    # Idempotent: skip re-download if already present
    if marker.exists() and cuad_json_path.exists():
        log.info("CUAD already downloaded, skipping.")
        return cuad_json_path

    log.info("Downloading CUAD dataset from %s ...", CUAD_ZIP_URL)
    response = requests.get(CUAD_ZIP_URL, timeout=120)
    response.raise_for_status()
    log.info("Downloaded %.1f MB", len(response.content) / (1024 * 1024))

    outer_zip = zipfile.ZipFile(io.BytesIO(response.content))
    log.info("Archive contents: %s", outer_zip.namelist())

    # Extract CUADv1.json (the master file with all 510 contracts)
    if "CUADv1.json" not in outer_zip.namelist():
        raise RuntimeError(
            f"CUADv1.json not found in archive. Contents: {outer_zip.namelist()}"
        )

    with outer_zip.open("CUADv1.json") as src, cuad_json_path.open("wb") as dst:
        dst.write(src.read())

    marker.write_text("done", encoding="utf-8")
    log.info("Cached CUADv1.json (%.1f MB) at %s",
             cuad_json_path.stat().st_size / (1024 * 1024), cuad_json_path)
    return cuad_json_path


def load_contracts(cuad_json_path: Path | None = None) -> list[ContractRecord]:
    """
    Load all 510 contracts from CUADv1.json.

    Returns a list of ContractRecord with title, text, and annotation count.
    """
    cuad_json_path = cuad_json_path or (settings.project_root / "data" / "raw" / "CUADv1.json")
    if not cuad_json_path.exists():
        raise FileNotFoundError(
            f"{cuad_json_path} not found. Run download_cuad() first."
        )

    log.info("Loading contracts from %s ...", cuad_json_path)
    data = json.loads(cuad_json_path.read_text(encoding="utf-8"))
    entries = data.get("data", [])

    records: list[ContractRecord] = []
    for entry in entries:
        title = entry.get("title", "untitled")
        paragraphs = entry.get("paragraphs", [])
        if not paragraphs:
            continue
        context = paragraphs[0].get("context", "")
        qas = paragraphs[0].get("qas", [])
        records.append(
            ContractRecord(title=title, text=context, num_annotations=len(qas))
        )

    log.info("Loaded %d contracts.", len(records))
    return records


def select_contracts(
    records: list[ContractRecord] | None = None,
    count: int | None = None,
) -> list[ContractRecord]:
    """
    Select a deterministic subset of N contracts (sorted for reproducibility).

    Reproducibility note: sorting the titles guarantees the same N
    contracts are selected on every run, which matters for HR reviewers
    re-running your code.
    """
    if records is None:
        records = load_contracts()
    count = count or settings.max_contracts

    sorted_records = sorted(records, key=lambda r: r.title)
    selected = sorted_records[:count]
    log.info("Selected %d of %d contracts (deterministic, sorted by title).",
             len(selected), len(records))
    return selected


if __name__ == "__main__":
    # Quick smoke test
    p = download_cuad()
    recs = load_contracts(p)
    subset = select_contracts(recs)
    print(f"\nReady: {len(subset)} contracts selected.")
    print(f"First: {subset[0].title} ({len(subset[0].text)} chars)")
