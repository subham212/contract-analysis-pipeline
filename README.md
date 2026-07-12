# 📜 Contract Analysis Pipeline (CUAD)

An LLM-powered pipeline that reads legal contracts, extracts key clauses, and generates concise summaries — built on the [CUAD dataset](https://www.atticusprojectai.org/cuad).

> **What it does:** Give it 50 contracts → it gives you back a clean CSV/JSON with the **termination**, **confidentiality**, and **liability** clauses of each, plus a 100–150 word summary.

> **Data note:** CUAD ships as a SQuAD-format JSON (`CUADv1.json`) where each contract's full text is already provided in the `context` field. The pipeline reads this text directly. A standalone `pdf_extractor.py` module is included to demonstrate the PDF-text-extraction capability for arbitrary new PDFs.

---

## ✨ Features

- 📄 **PDF → Text** extraction with cleanup (hyphenation, page numbers, unicode)
- 🧠 **LLM extraction** of 3 clause types (termination, confidentiality, liability)
- ✍️ **Concise summaries** (100–150 words: purpose, obligations, risks)
- 🔁 **Provider failover**: Gemini (primary) → OpenAI-compatible (fallback) automatically
- 🧩 **Token-safe chunking** for long contracts (30–80 pages)
- 📊 **Dual output**: CSV (analysts) + JSON (programmatic)
- 🔎 **Bonus — Semantic search** over clauses via embeddings (RAG-style)
- 📚 **Bonus — Few-shot prompting** mode for higher extraction accuracy
- 🧪 **Unit tests** included
- 🔒 **Secure by design**: API keys never leave your `.env`

---

## 🏗️ Architecture / Flow Diagram

```
 ┌─────────────┐
 │  CUAD ZIP   │  (download once, ~18 MB, cached)
 │  CUADv1.json│  510 contracts (SQuAD-format JSON)
 └──────┬──────┘
        │ select 50 (deterministic, sorted by title)
        ▼
 ┌─────────────────┐
 │ ContractRecord  │  title + full contract text (context)
 │ (text in memory)│
 └────────┬────────┘
          │ clean text (already extracted by The Atticus Project;
          │           PDF extractor available for new PDFs)
          ▼
 ┌────────────────────┐
 │   chunk_text()     │  sliding window with overlap
 │   (token-safe)     │  (~6k tokens/chunk)
 └────────┬───────────┘
          │ chunks
          ▼
               ┌─────────────────────────────┐
               │   LLM Router (failover)      │
               │   ┌─────────────────────┐    │
               │   │ 1. Gemini 1.5 Flash │────┼── quota? → next
               │   └─────────────────────┘    │
               │   ┌─────────────────────┐    │
               │   │ 2. OpenAI-compat    │────┼── try here
               │   └─────────────────────┘    │
               └──────────┬──────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Clause Extraction│    │   Summarization  │
    │ (termination,    │    │   100–150 words  │
    │  confidentiality,│    │                  │
    │  liability)      │    │                  │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
              ┌─────────────────────┐
              │  ContractResult     │  one per contract
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌──────────────┐
   │  CSV out   │ │  JSON out  │ │ Semantic idx │
   │ (analysts) │ │ (programs) │ │  (bonus)     │
   └────────────┘ └────────────┘ └──────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- A Google AI Studio API key (free) — [get one here](https://aistudio.google.com/app/apikey)

### 2. Install
```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure
```bash
# Copy the template and fill in your keys
cp .env.example .env
```
Edit `.env` and set at least:
```
GEMINI_API_KEY=your_key_here
```
(Optional: also set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` for fallback.)

### 4. Run
```bash
# Full pipeline (50 contracts)
python main.py run

# Quick test (5 contracts)
python main.py run --max 5

# Semantic search over results
python main.py search "termination for cause"
```

Results land in `/output`:
- `latest.csv` — analyst-friendly spreadsheet
- `latest.json` — structured data with metadata
- `semantic_index.json` — clause embeddings (bonus)

---

## 📁 Project Structure

```
Contract-AI-Pipeline/
├── main.py                  # 🚪 Entry point (CLI)
├── requirements.txt         # 📦 Dependencies
├── .env.example             # 🔑 Config template (copy to .env)
├── .gitignore               # 🛡️ Protects secrets + data
├── README.md                # 📖 This file
├── src/
│   ├── config.py            # ⚙️  Settings loader (.env → typed)
│   ├── logger.py            # 📝 Centralized logging
│   ├── data_loader.py       # ⬇️  CUAD download + subset selection
│   ├── pdf_extractor.py     # 📄 PDF → clean text
│   ├── chunker.py           # 🧩 Token-safe chunking
│   ├── llm_client.py        # 🧠 Gemini + OpenAI failover router
│   ├── prompts.py           # 💬 Zero-shot + few-shot prompts
│   ├── pipeline.py          # 🔄 Orchestrator (ties it all together)
│   ├── output_writer.py     # 📊 CSV + JSON writers
│   └── semantic_search.py   # 🔎 Bonus: embedding-based search
├── tests/
│   └── test_modules.py      # 🧪 Unit tests
├── data/
│   └── raw/                 # (CUAD PDFs download here — gitignored)
└── output/                  # (results land here — gitignored)
```

---

## 🧠 Approach & Design Decisions

### Why chunking even with Gemini's 1M context?
Gemini 1.5 Flash can technically hold a whole contract, but:
1. **Portability** — the pipeline must work on *any* provider (OpenAI, Llama, etc.)
2. **Cost** — extraction over 6k-token chunks is cheaper than one giant call
3. **Accuracy** — focused chunks reduce "lost in the middle" attention degradation

### Why provider failover?
Free tiers have quotas. When Gemini's quota runs out mid-run, the pipeline **automatically** switches to the OpenAI-compatible endpoint instead of crashing. Production systems must degrade gracefully.

### Why JSON-strict prompts?
Deterministic parsing. Free-text answers would require fragile regex. Forcing the LLM to return `{"termination": "...", ...}` makes downstream code reliable.

### Why overlap in chunks?
A clause might span a chunk boundary (start in chunk 2, end in chunk 3). Overlap (400 tokens) ensures the full clause appears in at least one chunk intact.

### Why deterministic contract selection?
Sorting filenames before taking the first N guarantees the **same 50 contracts** on every run — essential for reproducibility and for HR reviewers re-running your code.

---

## 🧪 Testing

```bash
# Run unit tests (no API keys needed)
python tests/test_modules.py
# OR with pytest
python -m pytest tests/ -v
```

Tests cover: chunking, text normalization, prompt assembly, and the result dataclass.

---

## 🔒 Security Notes

- API keys live in `.env` only — **gitignored**, never committed.
- `.env.example` ships with empty values as a template.
- No keys are logged or printed at any log level.

---

## 📚 Dataset

**CUAD** (Contract Understanding Atticus Dataset), NeurIPS 2021.
- 510 commercial legal contracts
- 13,000+ human annotations across 41 clause types
- Source: [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad)

---

## ⚠️ Limitations & Future Work

- **OCR**: scanned/image-only PDFs would need an OCR pass (e.g. Tesseract). CUAD PDFs are text-based, so this pipeline extracts directly.
- **Eval metrics**: for production, compare extracted clauses against CUAD's ground-truth annotations to compute precision/recall/F1.
- **Streaming UI**: a web front-end (Streamlit/Gradio) for interactive upload + search would make this a full product.
- **Vector DB**: for scale, replace the in-memory index with FAISS, Pinecone, or Weaviate.

---

## 📄 License

This project is for educational/portfolio purposes. The CUAD dataset is released under its own license (see source).
