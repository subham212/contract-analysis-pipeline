# 📜 Contract Analysis & Legal RAG Pipeline (CUAD)

[![Live Demo](https://img.shields.io/badge/Live_Demo-legal--contract--pipeline--ynop.onrender.com-0058be?style=for-the-badge&logo=render)](https://legal-contract-pipeline-ynop.onrender.com/)

A production-grade, LLM-powered contract intelligence system that reads commercial contracts, extracts key legal clauses, and powers an interactive AI Legal Assistant with verifiable verbatim citations — built on the [CUAD dataset](https://www.atticusprojectai.org/cuad) (510 contracts, 13,000+ annotations).

---

## Features

- **Title-Based Legal Chunking**: Document-aware chunker that partitions contracts by natural legal boundaries (`ARTICLE`, `SECTION`, `CLAUSE`, numbered headings, and exhibits).
- **Primary LLM Engine**: Powered by **ZenMux** (`deepseek/deepseek-v4.1-flash`) via an OpenAI-compatible gateway for high speed and low latency.
- **Neural & Offline Embeddings**: Vectorized with **ZenMux** (`openai/text-embedding-3-small` at 768-D) with automatic fallback to a local deterministic hash vectorizer.
- **Dual-Representation RAG ("Way B" Architecture)**:
  - **Vector Search on Summaries**: Indexes a concise 2–3 sentence legal summary for maximum vector retrieval accuracy.
  - **Generation from Verbatim Text**: Pulls the **100% original, unedited raw contract text** into the prompt, eliminating hallucinations.
- **Automated Clause Extraction**: Extracts **Termination**, **Confidentiality**, and **Liability** clauses into structured JSON & CSV formats.
- **Interactive Web UI**: Modern FastAPI application featuring:
  - **Browse Contracts**: Filter and inspect extracted clauses across contract portfolios.
  - **AI Legal Assistant**: Ask legal questions with expandable verbatim source citations.
- **16 Passing Unit Tests**: Fully validated suite covering chunking, normalization, embeddings, ChromaDB, and FastAPI endpoints.
- **Zero-Leak Security**: Secrets live strictly in `.env` (git-ignored); zero API keys are ever logged.

---

## 🏗️ Architecture / Flow Diagram

```
 ┌──────────────────────────────────────────────────────────┐
 │                     Raw Contracts                        │
 │   CUADv1.json (Dataset)  /  PDFs (via pdfplumber)        │
 └────────────────────────────┬─────────────────────────────┘
                              │ clean text
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │      Title-Based Chunker (src/rag/section_chunker.py)     │
 │   Splits by: ARTICLE / SECTION / CLAUSE / 1. Title       │
 └─────────────┬──────────────────────────────┬─────────────┘
               │                              │
     (A) Verbatim Raw Excerpt       (B) Search Summarizer
      saved in metadata["raw_content"]   deepseek-v4.1-flash (cached)
               │                              │
               │                              ▼
               │                    ┌───────────────────┐
               │                    │ Vector Embedding  │
               │                    │ text-embedding-   │
               │                    │ 3-small (768-D)   │
               │                    └─────────┬─────────┘
               │                              │
               ▼                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │               Persistent ChromaDB ("Way B")              │
 │   page_content = Summary  |  metadata = Raw Legal Text   │
 └────────────────────────────┬─────────────────────────────┘
                              │
               User Query: "What is notice period?"
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │                  RAG Execution Engine                    │
 │  1. Vector search finds top-k matching section summaries │
 │  2. Retrieves verbatim raw legal excerpts from metadata  │
 │  3. Prompts DeepSeek with strict anti-hallucination rules│
 │  4. Generates synthesized answer with source citations   │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │              Web UI & CLI Presentation                   │
 │   • Synthesized legal answer                             │
 │   • Clickable, expandable verbatim citation accordions   │
 └──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- A [ZenMux](https://zenmux.ai) API key (or any OpenAI-compatible provider)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Contract-AI-Pipeline.git
cd Contract-AI-Pipeline

# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash)
source .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your ZenMux API settings in `.env`:
```ini
# ZenMux Gateway
OPENAI_BASE_URL=https://zenmux.ai/api/v1
OPENAI_MODEL=deepseek/deepseek-v4.1-flash
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENAI_API_KEY=your_zenmux_api_key_here

# Pipeline Settings
MAX_CONTRACTS=50
OUTPUT_DIR=output
EXTRACTION_MODE=zero_shot
```

---

## 💻 CLI & Web Usage

### Ask Legal Questions (RAG Engine)
Query indexed contracts with grounded citations:
```bash
python main.py ask "What is the notice period required to terminate without cause?"
```

### Index Contracts for RAG
Index contracts into the persistent ChromaDB vector store:
```bash
python main.py index-rag --max 10
```

### Launch Interactive Web Application
Start the FastAPI server on port 7860:
```bash
python -m uvicorn app:app --port 7860
```
Open your browser at `http://localhost:7860`:
* **Browse Contracts**: Inspect extracted clauses.
* **AI Legal Assistant**: Ask multi-contract legal questions with verified citations.

### Run Full Extraction Pipeline (50 Contracts)
Extract termination, confidentiality, and liability clauses into CSV & JSON:
```bash
python main.py run --max 50
# Or run in batches:
python run_all.py 50 5 0
```

---

## 📁 Repository Structure

```
Contract-AI-Pipeline/
├── main.py                     # CLI entry point (run, ask, index-rag, merge)
├── app.py                      # FastAPI Web UI + AI Legal Assistant
├── run_all.py                  # Batch execution script
├── requirements.txt            # Minimal dependencies (openai, chromadb, fastapi, pdfplumber)
├── Dockerfile                  # Container deployment setup (Hugging Face / Render)
├── .env.example                # Configuration template
├── .gitignore                  # Prevents committing secrets, data, and ChromaDB binaries
├── README.md                   # System documentation
├── src/
│   ├── config.py               # Typed settings loader via python-dotenv
│   ├── logger.py               # UTF-8 safe logging configuration
│   ├── data_loader.py          # CUAD dataset downloader and parser
│   ├── pdf_extractor.py        # PDF text extractor with layout normalization
│   ├── chunker.py              # Token-safe sliding window chunker (for extraction)
│   ├── llm_client.py           # Unified ZenMux/OpenAI router with pacing
│   ├── prompts.py              # Zero-shot and few-shot extraction prompts
│   ├── pipeline.py             # Extraction orchestrator
│   ├── output_writer.py        # CSV/JSON exporters
│   └── rag/                    # Advanced Legal RAG Engine
│       ├── section_chunker.py  # Title/heading-based contract partitioner
│       ├── summarizer.py       # Search summary generator with disk cache
│       ├── vector_store.py     # ChromaDB persistent store ("Way B" architecture)
│       └── engine.py           # Grounded Q&A engine with citations
├── tests/
│   └── test_modules.py         # 16 unit tests (chunker, embeddings, ChromaDB, RAG, API)
└── output/                     # Deliverables (CSV, JSON, ChromaDB index - gitignored)
```

---

## 🧠 Design Philosophy

### 1. Title-Based Section Chunking vs. Fixed Token Windows
Fixed token windows (e.g. cutting arbitrarily every 500 characters) frequently bisect legal clauses and discard critical heading hierarchy. Our [section_chunker.py](file:///e:/Contract-AI-Pipeline/src/rag/section_chunker.py) detects formal legal headings (`ARTICLE`, `SECTION`, `1. Definitions`), maintaining each section as a cohesive legal entity.

### 2. Dual-Representation "Way B" Vector Store
* **Search Matching:** Long legal clauses contain dense boilerplate that degrades vector matching accuracy. We vectorize concise 2–3 sentence summaries.
* **Grounded Generation:** When retrieving sources, the engine extracts the **verbatim raw legal excerpt** stored in document metadata. The LLM answers strictly from original legal prose, eliminating hallucinations.

### 3. Disk-Cached Summarization
Generating section summaries can be expensive. [summarizer.py](file:///e:/Contract-AI-Pipeline/src/rag/summarizer.py) caches generated summaries on disk in `output/rag_summaries_cache.json`. Subsequent queries or re-indexes run instantly at $0 cost.

---

## 🧪 Running Tests

Run the full standalone test suite:
```bash
python tests/test_modules.py
```

Output:
```
  ✓ test_app_rag_endpoint
  ✓ test_chromadb_way_b
  ✓ test_chunker_long_text
  ✓ test_chunker_short_text
  ✓ test_contract_embeddings
  ✓ test_contract_result_dataclass
  ✓ test_estimate_tokens
  ✓ test_extraction_prompt_few_shot
  ✓ test_extraction_prompt_zero_shot
  ✓ test_normalize_hyphenation
  ✓ test_normalize_page_numbers
  ✓ test_normalize_whitespace
  ✓ test_rag_engine_end_to_end
  ✓ test_section_chunker
  ✓ test_summarizer_cache
  ✓ test_summary_prompt

16 passed, 0 failed, 16 total.
```

---

## 📄 License
This project is open-source for educational and portfolio purposes. The CUAD dataset is provided by The Atticus Project under its respective license.
