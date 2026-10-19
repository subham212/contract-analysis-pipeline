"""
app.py — Custom FastAPI Web UI for the Contract Analysis Pipeline
=================================================================
Pixel-perfect Lexicon Precision Theme Implementation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import uvicorn

# ── Paths & Data ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CONTRACTS_JSON = ROOT / "output" / "final_contracts.json"

def _load_contracts() -> list[dict[str, Any]]:
    if not CONTRACTS_JSON.exists():
        return []
    data = json.loads(CONTRACTS_JSON.read_text(encoding="utf-8"))
    return data.get("results", [])

CONTRACTS: list[dict[str, Any]] = _load_contracts()
_CONTRACT_MAP: dict[str, dict[str, Any]] = {c["contract_id"]: c for c in CONTRACTS}

# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(title="Lexicon Precision API")

@app.get("/api/contracts")
def get_contracts() -> list[dict[str, str]]:
    """
    Return a lightweight list of contract IDs.
    Used to populate the frontend selection dropdown.
    """
    return [{"id": c["contract_id"]} for c in CONTRACTS]

@app.get("/api/contract/{contract_id}")
def get_contract(contract_id: str) -> dict[str, Any]:
    """
    Return the full extracted details and summary for a specific contract.
    """
    if contract_id not in _CONTRACT_MAP:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    c = _CONTRACT_MAP[contract_id]
    summary_text = c.get("summary", "No summary available.")
    return {
        "id": c["contract_id"],
        "summary": summary_text,
        "word_count": len(summary_text.split()),
        "char_count": c.get('char_count', 0),
        "provider": c.get('provider', 'N/A'),
        "termination_clause": c.get("termination_clause", "Not found"),
        "confidentiality_clause": c.get("confidentiality_clause", "Not found"),
        "liability_clause": c.get("liability_clause", "Not found")
    }


# ── RAG Endpoint ─────────────────────────────────────────────────────
RAG_ENGINE = None

def _get_rag_engine():
    global RAG_ENGINE
    if RAG_ENGINE is None:
        from src.rag.engine import ContractRAGEngine
        RAG_ENGINE = ContractRAGEngine()
    return RAG_ENGINE

@app.post("/api/rag/ask")
def api_rag_ask(payload: dict[str, Any]) -> dict[str, Any]:
    """Ask an arbitrary question across indexed contracts with citation backing."""
    q = (payload.get("query") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    top_k = int(payload.get("top_k", 3))
    contract_id = payload.get("contract_id") or None
    engine = _get_rag_engine()

    # If ChromaDB is empty on first boot, index initial contracts
    if engine.vector_store.count() == 0 and CONTRACTS:
        for c in CONTRACTS[:5]:
            synth = (
                f"CONTRACT: {c['contract_id']}\n\n"
                f"EXECUTIVE SUMMARY:\n{c.get('summary', '')}\n\n"
                f"TERMINATION:\n{c.get('termination_clause', '')}\n\n"
                f"CONFIDENTIALITY:\n{c.get('confidentiality_clause', '')}\n\n"
                f"LIABILITY:\n{c.get('liability_clause', '')}"
            )
            engine.index_contract(c["contract_id"], synth)

    resp = engine.ask(q, top_k=top_k, filter_contract=contract_id)
    return {
        "query": resp.query,
        "answer": resp.answer,
        "provider": resp.provider,
        "model": resp.model,
        "citations": [
            {
                "contract_title": c.contract_title,
                "section_title": c.section_title,
                "score": round(c.score * 100, 1),
                "raw_excerpt": c.raw_excerpt,
                "summary": c.summary,
            }
            for c in resp.citations
        ]
    }

@app.get("/api/rag/ask")
def api_rag_ask_get(q: str = Query(..., min_length=1), top_k: int = Query(3, ge=1, le=10)):
    return api_rag_ask({"query": q, "top_k": top_k})


# ── Frontend HTML/CSS/JS ─────────────────────────────────────────────
HTML_CONTENT = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"><link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=block" rel="stylesheet"><script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script><script id="tailwind-config">try{
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    "colors": {
                        "primary-fixed": "#dae2fd",
                        "on-error-container": "#93000a",
                        "primary-fixed-dim": "#bec6e0",
                        "tertiary-fixed-dim": "#dec29a",
                        "on-surface-variant": "#45464d",
                        "on-surface": "#1b1b1d",
                        "error": "#ba1a1a",
                        "tertiary": "#000000",
                        "secondary-fixed": "#d8e2ff",
                        "on-tertiary-fixed": "#271901",
                        "on-secondary-fixed-variant": "#004395",
                        "surface-container-lowest": "#ffffff",
                        "background": "#fcf8fa",
                        "surface-container-low": "#f6f3f5",
                        "tertiary-fixed": "#fcdeb5",
                        "surface-container-highest": "#e4e2e4",
                        "on-tertiary": "#ffffff",
                        "inverse-surface": "#303032",
                        "surface-container": "#f0edef",
                        "on-primary-fixed": "#131b2e",
                        "tertiary-container": "#271901",
                        "surface-variant": "#e4e2e4",
                        "on-primary-fixed-variant": "#3f465c",
                        "outline": "#76777d",
                        "outline-variant": "#c6c6cd",
                        "surface-dim": "#dcd9db",
                        "error-container": "#ffdad6",
                        "secondary-container": "#2170e4",
                        "secondary": "#0058be",
                        "on-error": "#ffffff",
                        "on-tertiary-fixed-variant": "#574425",
                        "inverse-primary": "#bec6e0",
                        "on-primary-container": "#7c839b",
                        "secondary-fixed-dim": "#adc6ff",
                        "surface": "#fcf8fa",
                        "surface-bright": "#fcf8fa",
                        "primary": "#000000",
                        "surface-tint": "#565e74",
                        "on-secondary-fixed": "#001a42",
                        "on-secondary": "#ffffff",
                        "on-secondary-container": "#fefcff",
                        "on-tertiary-container": "#98805d",
                        "inverse-on-surface": "#f3f0f2",
                        "primary-container": "#131b2e",
                        "on-primary": "#ffffff",
                        "surface-container-high": "#eae7e9",
                        "on-background": "#1b1b1d"
                    },
                    "borderRadius": {
                        "DEFAULT": "0.25rem",
                        "lg": "0.5rem",
                        "xl": "0.75rem",
                        "full": "9999px"
                    },
                    "spacing": {
                        "base": "8px",
                        "container-max": "1440px",
                        "gutter": "24px",
                        "margin-mobile": "16px",
                        "margin-desktop": "40px"
                    },
                    "fontFamily": {
                        "headline-lg": ["Inter"],
                        "body-lg": ["Inter"],
                        "headline-md": ["Inter"],
                        "label-md": ["Inter"],
                        "label-sm": ["Inter"],
                        "display-lg": ["Inter"],
                        "body-md": ["Inter"]
                    },
                    "fontSize": {
                        "headline-lg": ["32px", {"lineHeight": "40px", "letterSpacing": "-0.01em", "fontWeight": "600"}],
                        "body-lg": ["18px", {"lineHeight": "32px", "fontWeight": "400"}],
                        "headline-md": ["24px", {"lineHeight": "32px", "fontWeight": "600"}],
                        "label-md": ["14px", {"lineHeight": "20px", "letterSpacing": "0.01em", "fontWeight": "500"}],
                        "label-sm": ["12px", {"lineHeight": "16px", "fontWeight": "600"}],
                        "display-lg": ["48px", {"lineHeight": "56px", "letterSpacing": "-0.02em", "fontWeight": "700"}],
                        "body-md": ["16px", {"lineHeight": "28px", "fontWeight": "400"}]
                    }
                },
            },
        }
    }catch(_e){}</script><meta charset="utf-8">
    <style>
        .view-content { display: none; opacity: 0; transition: opacity 0.3s ease; }
        .view-content.active { display: block; opacity: 1; }
        .legal-scroll { scrollbar-width: thin; scrollbar-color: var(--outline-variant) transparent; }
        .legal-scroll::-webkit-scrollbar { width: 6px; }
        .legal-scroll::-webkit-scrollbar-thumb { background-color: var(--outline-variant); border-radius: 10px; }
    </style>
</head><body class="bg-background text-on-surface">

<!-- SIDE NAVIGATION BAR -->
<aside class="hidden md:flex h-screen w-72 fixed left-0 top-0 bg-surface-container-low/90 backdrop-blur-md border-r border-outline-variant/20 shadow-md flex-col p-6 space-y-4 z-40">
    <div class="flex items-center space-x-3 mb-8">
        <div class="w-10 h-10 bg-primary-container rounded-lg flex items-center justify-center">
            <span class="material-symbols-outlined text-on-primary" style="font-variation-settings: 'FILL' 1;">analytics</span>
        </div>
        <div>
            <h1 class="font-headline-md text-headline-md text-primary leading-tight">Contract Analysis</h1>
            <p class="font-label-sm text-label-sm text-on-surface-variant opacity-70">LLM-powered extraction</p>
        </div>
    </div>
    
    <nav class="flex-1 space-y-2">
        <button onclick="switchView('browse')" id="nav-browse" class="w-full flex items-center space-x-3 p-3 bg-secondary text-on-secondary rounded-lg font-bold transition-all duration-200">
            <span class="material-symbols-outlined">folder_open</span>
            <span class="font-label-md text-label-md">Browse Contracts</span>
        </button>
        <button onclick="switchView('rag')" id="nav-rag" class="w-full flex items-center space-x-3 p-3 text-on-surface-variant hover:bg-surface-container-high rounded-lg transition-transform duration-200 hover:translate-x-1">
            <span class="material-symbols-outlined">psychology</span>
            <span class="font-label-md text-label-md">AI Legal Assistant</span>
        </button>
    </nav>
</aside>

<!-- MAIN CONTENT AREA -->
<main class="md:ml-72 min-h-screen">
    <!-- TOP APP BAR -->
    <header class="sticky top-0 z-30 flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop h-20 bg-surface/80 backdrop-blur-xl border-b border-outline-variant/30 shadow-sm max-w-container-max mx-auto">
        <div class="flex flex-col min-w-0">
            <h2 class="font-headline-md text-lg md:text-headline-md font-bold text-primary truncate">Contract Analysis Pipeline</h2>
            <p class="font-label-sm text-xs md:text-label-sm text-on-surface-variant truncate">LLM-powered clause extraction and summarization</p>
        </div>
    </header>

    <!-- CONTENT CANVAS -->
    <div class="max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop py-4 md:py-8 space-y-gutter">
        
        <!-- MOBILE NAVIGATION TOGGLE -->
        <div class="flex md:hidden justify-center mb-2">
            <div class="flex bg-surface-container-high rounded-xl p-1 shadow-inner w-full max-w-sm">
                <button onclick="switchView('browse')" id="nav-browse-mobile" class="flex-1 py-2 bg-surface-container-lowest text-primary rounded-lg font-bold text-sm shadow-sm transition-colors text-center">Browse</button>
                <button onclick="switchView('rag')" id="nav-rag-mobile" class="flex-1 py-2 text-on-surface-variant rounded-lg font-bold text-sm transition-colors text-center">Ask AI</button>
            </div>
        </div>
        
        <!-- BROWSE VIEW -->
        <div id="view-browse" class="view-content active">
            <!-- CONTRACT SELECT DROPDOWN -->
            <section class="w-full mb-8">
                <div class="relative w-full max-w-2xl mx-auto group">
                    <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                        <span class="material-symbols-outlined text-secondary">description</span>
                    </div>
                    <select id="contract-select" onchange="loadContract(this.value)" class="block w-full pl-12 pr-10 py-4 bg-surface-container-lowest border border-outline-variant/50 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-secondary/10 focus:border-secondary font-body-md text-body-md transition-all cursor-pointer truncate">
                        <option value="">Select a contract...</option>
                    </select>
                </div>
            </section>

            <!-- DASHBOARD GRID -->
            <div id="contract-display">
                <div class="text-center text-on-surface-variant p-20 border border-outline-variant/30 rounded-xl bg-surface-container-lowest shadow-sm">
                    Select a contract from the dropdown above to view its analysis.
                </div>
            </div>
        </div>

        <!-- RAG VIEW -->
        <div id="view-rag" class="view-content">
            <div class="w-full mb-6 max-w-4xl mx-auto">
                <div class="flex flex-col md:flex-row gap-4">
                    <div class="relative flex-1">
                        <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                            <span class="material-symbols-outlined text-secondary">psychology</span>
                        </div>
                        <input type="text" id="rag-query" placeholder="Ask any question across contracts (e.g., notice required to terminate?)..." onkeypress="handleRAGEnter(event)" class="block w-full pl-12 pr-4 py-4 bg-surface-container-lowest border border-outline-variant/50 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-secondary/10 focus:border-secondary font-body-md text-body-md transition-all">
                    </div>
                    <div class="flex items-center gap-4">
                        <label for="rag-top-k" class="font-label-md text-on-surface-variant whitespace-nowrap">Sources:</label>
                        <input type="number" id="rag-top-k" value="3" min="1" max="10" class="w-20 px-4 py-4 bg-surface-container-lowest border border-outline-variant/50 rounded-xl shadow-sm text-center">
                        <button onclick="askRAG()" id="rag-btn" class="px-8 py-4 bg-secondary text-on-secondary rounded-xl font-bold hover:bg-secondary-container transition-colors shadow-sm flex items-center gap-2">
                            <span class="material-symbols-outlined text-sm">auto_awesome</span> Ask AI
                        </button>
                    </div>
                </div>

                <!-- SUGGESTED QUERIES -->
                <div class="flex flex-wrap gap-2 mt-4 items-center">
                    <span class="font-label-sm text-xs text-on-surface-variant opacity-80">Suggested:</span>
                    <button onclick="setRAGQuery('What is the notice period required to terminate without cause?')" class="px-3 py-1 bg-surface-container-high hover:bg-surface-container-highest text-on-surface-variant rounded-full font-label-sm text-xs transition-colors">Notice to terminate?</button>
                    <button onclick="setRAGQuery('What are the limitations of liability and damage caps?')" class="px-3 py-1 bg-surface-container-high hover:bg-surface-container-highest text-on-surface-variant rounded-full font-label-sm text-xs transition-colors">Liability caps?</button>
                    <button onclick="setRAGQuery('What information is excluded from confidentiality obligations?')" class="px-3 py-1 bg-surface-container-high hover:bg-surface-container-highest text-on-surface-variant rounded-full font-label-sm text-xs transition-colors">Confidentiality exclusions?</button>
                </div>
            </div>

            <div id="rag-results" class="space-y-6 max-w-4xl mx-auto">
                <div class="text-center text-on-surface-variant p-20 border border-outline-variant/30 rounded-xl bg-surface-container-lowest shadow-sm">
                    Ask any legal or operational question above. The AI will retrieve the exact contract sections and synthesize a citation-backed answer.
                </div>
            </div>
        </div>

    </div>
</main>

<script>
    function switchView(viewName) {
        document.querySelectorAll('.view-content').forEach(el => el.classList.remove('active'));
        const targetView = document.getElementById(`view-${viewName}`);
        if (targetView) targetView.classList.add('active');
        
        const navBrowse = document.getElementById('nav-browse');
        const navRag = document.getElementById('nav-rag');
        const navBrowseMob = document.getElementById('nav-browse-mobile');
        const navRagMob = document.getElementById('nav-rag-mobile');
        
        const activeDesktop = 'w-full flex items-center space-x-3 p-3 bg-secondary text-on-secondary rounded-lg font-bold transition-all duration-200';
        const inactiveDesktop = 'w-full flex items-center space-x-3 p-3 text-on-surface-variant hover:bg-surface-container-high rounded-lg transition-transform duration-200 hover:translate-x-1';
        
        const activeMobile = 'flex-1 py-2 bg-surface-container-lowest text-primary rounded-lg font-bold text-sm shadow-sm transition-colors text-center';
        const inactiveMobile = 'flex-1 py-2 text-on-surface-variant rounded-lg font-bold text-sm transition-colors text-center';

        [navBrowse, navRag].forEach(el => { if(el) el.className = inactiveDesktop; });
        [navBrowseMob, navRagMob].forEach(el => { if(el) el.className = inactiveMobile; });

        if (viewName === 'browse') {
            if (navBrowse) navBrowse.className = activeDesktop;
            if (navBrowseMob) navBrowseMob.className = activeMobile;
        } else if (viewName === 'rag') {
            if (navRag) navRag.className = activeDesktop;
            if (navRagMob) navRagMob.className = activeMobile;
        }
    }

    async function init() {
        try {
            const res = await fetch('/api/contracts');
            const data = await res.json();
            const select = document.getElementById('contract-select');
            
            data.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.id;
                opt.textContent = c.id;
                select.appendChild(opt);
            });
            // Do not auto-load the first contract; wait for user selection.
        } catch (e) { console.error(e); }
    }

    async function loadContract(id) {
        const display = document.getElementById('contract-display');
        if (!id) {
            display.innerHTML = `<div class="text-center text-on-surface-variant p-20 border border-outline-variant/30 rounded-xl bg-surface-container-lowest shadow-sm">Select a contract from the dropdown above to view its analysis.</div>`;
            return;
        }

        display.innerHTML = `<div class="text-center text-on-surface-variant p-20 border border-outline-variant/30 rounded-xl bg-surface-container-lowest shadow-sm">Loading analysis...</div>`;
        
        try {
            const res = await fetch(`/api/contract/${encodeURIComponent(id)}`);
            const c = await res.json();

            display.innerHTML = `
                <div class="grid grid-cols-12 gap-gutter">
                    <!-- SUMMARY CARD -->
                    <article class="col-span-12 bg-surface-container-lowest border border-outline-variant/30 rounded-xl shadow-sm overflow-hidden flex flex-col">
                        <header class="px-6 py-4 bg-surface-container-low/50 backdrop-blur-sm border-b border-outline-variant/10 flex flex-col md:flex-row md:justify-between items-start md:items-center gap-4">
                            <div class="flex items-center space-x-2">
                                <span class="material-symbols-outlined text-secondary" style="font-variation-settings: 'FILL' 1;">summarize</span>
                                <h3 class="font-headline-md text-xl md:text-headline-md text-on-surface">Executive Summary</h3>
                            </div>
                            <div class="flex flex-row flex-nowrap gap-1 md:gap-2 items-center overflow-x-auto no-scrollbar w-full">
                                <span class="px-1.5 md:px-3 py-0.5 md:py-1 bg-surface-container-high rounded-full font-label-sm text-[8px] md:text-label-sm text-on-surface-variant whitespace-nowrap">Word count: ${c.word_count.toLocaleString()}</span>
                                <span class="px-1.5 md:px-3 py-0.5 md:py-1 bg-surface-container-high rounded-full font-label-sm text-[8px] md:text-label-sm text-on-surface-variant whitespace-nowrap">Characters: ${c.char_count.toLocaleString()}</span>
                                <span class="px-1.5 md:px-3 py-0.5 md:py-1 bg-primary-fixed text-on-primary-fixed rounded-full font-label-sm text-[8px] md:text-label-sm flex items-center gap-1 whitespace-nowrap">
                                    <span class="material-symbols-outlined text-[8px] md:text-xs">auto_awesome</span> ${c.provider}
                                </span>
                            </div>
                        </header>
                        <div class="p-4 md:p-8 max-w-[900px] mx-auto overflow-y-auto legal-scroll max-h-48 md:max-h-none">
                            <p class="font-body-lg text-sm md:text-body-lg text-on-surface-variant leading-relaxed">${c.summary}</p>
                        </div>
                    </article>

                    <!-- MIDDLE ROW: TERMINATION & CONFIDENTIALITY -->
                    <article class="col-span-6 md:col-span-6 bg-error-container/20 border border-error/10 rounded-xl shadow-sm overflow-hidden flex flex-col h-48 md:h-[400px]">
                        <header class="px-2 md:px-6 py-2 md:py-4 bg-error-container/30 border-b border-error/5 flex items-center space-x-1 md:space-x-2 truncate">
                            <span class="material-symbols-outlined text-error text-[16px] md:text-[24px]" style="font-variation-settings: 'FILL' 1;">cancel</span>
                            <h3 class="font-label-md text-[9px] md:text-label-md font-bold text-on-error-container uppercase tracking-wider truncate">Termination</h3>
                        </header>
                        <div class="p-2 md:p-6 overflow-y-auto legal-scroll flex-1 font-body-md text-[10px] md:text-body-md text-on-surface-variant">
                            <p class="">${c.termination_clause}</p>
                        </div>
                    </article>

                    <article class="col-span-6 md:col-span-6 bg-tertiary-fixed/10 border border-tertiary-fixed/30 rounded-xl shadow-sm overflow-hidden flex flex-col h-48 md:h-[400px]">
                        <header class="px-2 md:px-6 py-2 md:py-4 bg-tertiary-fixed/20 border-b border-tertiary-fixed/20 flex items-center space-x-1 md:space-x-2 truncate">
                            <span class="material-symbols-outlined text-on-tertiary-fixed-variant text-[16px] md:text-[24px]" style="font-variation-settings: 'FILL' 1;">lock</span>
                            <h3 class="font-label-md text-[9px] md:text-label-md font-bold text-on-tertiary-fixed-variant uppercase tracking-wider truncate">Confidentiality</h3>
                        </header>
                        <div class="p-2 md:p-6 overflow-y-auto legal-scroll flex-1 font-body-md text-[10px] md:text-body-md text-on-surface-variant">
                            <p class="">${c.confidentiality_clause}</p>
                        </div>
                    </article>

                    <!-- BOTTOM ROW: LIABILITY -->
                    <article class="col-span-12 bg-secondary-fixed/10 border border-secondary/10 rounded-xl shadow-sm overflow-hidden flex flex-col h-48 md:h-auto">
                        <header class="px-4 md:px-6 py-3 md:py-4 bg-secondary-fixed/20 border-b border-secondary/10 flex items-center space-x-2">
                            <span class="material-symbols-outlined text-secondary text-[18px] md:text-[24px]" style="font-variation-settings: 'FILL' 1;">gavel</span>
                            <h3 class="font-label-md text-[11px] md:text-label-md font-bold text-on-secondary-fixed-variant uppercase tracking-wider">Liability Clause</h3>
                        </header>
                        <div class="p-4 md:p-8 overflow-y-auto legal-scroll font-body-md text-[11px] md:text-body-md text-on-surface-variant max-w-[1000px]">
                            <p class="leading-relaxed">${c.liability_clause}</p>
                        </div>
                    </article>
                </div>
            `;
            
            // Re-bind hover events for micro-interactions
            document.querySelectorAll('article').forEach(card => {
                card.addEventListener('mouseenter', () => {
                    card.style.transform = 'translateY(-2px)';
                    card.style.transition = 'transform 0.2s ease-out';
                });
                card.addEventListener('mouseleave', () => {
                    card.style.transform = 'translateY(0px)';
                });
            });

        } catch (e) {
            display.innerHTML = `<div class="text-center text-error p-20 border border-outline-variant/30 rounded-xl bg-surface-container-lowest shadow-sm">Error loading contract.</div>`;
        }
    }

    function handleRAGEnter(e) { if (e.key === 'Enter') askRAG(); }
    function setRAGQuery(q) {
        document.getElementById('rag-query').value = q;
        askRAG();
    }

    async function askRAG() {
        const query = document.getElementById('rag-query').value.trim();
        const top_k = document.getElementById('rag-top-k').value || 3;
        const resultsDiv = document.getElementById('rag-results');
        const btn = document.getElementById('rag-btn');

        if (!query) return;
        btn.disabled = true;
        btn.innerHTML = `<span class="material-symbols-outlined text-sm animate-spin">refresh</span> Thinking...`;
        resultsDiv.innerHTML = `<div class="text-center text-on-surface-variant p-20 border border-outline-variant/30 rounded-xl bg-surface-container-lowest shadow-sm flex flex-col items-center gap-3">
            <span class="material-symbols-outlined text-secondary text-3xl animate-spin">refresh</span>
            <span>Searching contract sections and synthesizing answer...</span>
        </div>`;

        try {
            const res = await fetch('/api/rag/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, top_k: parseInt(top_k) })
            });
            const data = await res.json();
            
            let citationsHtml = '';
            if (data.citations && data.citations.length > 0) {
                data.citations.forEach((c, idx) => {
                    citationsHtml += `
                        <div class="border border-outline-variant/30 rounded-lg p-4 bg-surface-container-low/50">
                            <div class="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2 mb-2">
                                <div class="font-headline-md text-sm font-semibold text-primary truncate">
                                    [${idx+1}] ${c.contract_title} — ${c.section_title}
                                </div>
                                <span class="px-2.5 py-0.5 bg-secondary-fixed text-on-secondary-fixed-variant rounded-full text-xs font-label-sm w-fit">${c.score}% Match</span>
                            </div>
                            <details class="text-xs text-on-surface-variant cursor-pointer group">
                                <summary class="font-label-sm text-secondary hover:underline select-none">View Verbatim Contract Excerpt (Way B)</summary>
                                <div class="mt-2 p-3 bg-surface-container-lowest rounded border border-outline-variant/20 font-mono text-[11px] leading-relaxed max-h-48 overflow-y-auto legal-scroll whitespace-pre-wrap">${c.raw_excerpt}</div>
                            </details>
                        </div>
                    `;
                });
            }

            resultsDiv.innerHTML = `
                <article class="bg-surface-container-lowest border border-outline-variant/30 rounded-xl shadow-sm overflow-hidden">
                    <header class="px-6 py-4 bg-surface-container-low/60 border-b border-outline-variant/10 flex justify-between items-center">
                        <div class="flex items-center space-x-2">
                            <span class="material-symbols-outlined text-secondary" style="font-variation-settings: 'FILL' 1;">psychology</span>
                            <h3 class="font-headline-md text-base md:text-lg font-bold text-on-surface">Synthesized Answer</h3>
                        </div>
                        <span class="px-3 py-1 bg-primary-fixed text-on-primary-fixed rounded-full font-label-sm text-xs flex items-center gap-1">
                            <span class="material-symbols-outlined text-xs">auto_awesome</span> ${data.provider || 'ZenMux RAG'}
                        </span>
                    </header>
                    <div class="p-6 md:p-8 space-y-6">
                        <div class="font-body-lg text-sm md:text-base text-on-surface leading-relaxed whitespace-pre-line border-l-2 border-secondary pl-4">${data.answer}</div>
                        
                        <div>
                            <h4 class="font-label-md text-xs font-bold text-on-surface-variant uppercase tracking-wider mb-3">Verified Sources & Citations (${data.citations ? data.citations.length : 0})</h4>
                            <div class="space-y-3">${citationsHtml}</div>
                        </div>
                    </div>
                </article>
            `;
        } catch (e) {
            resultsDiv.innerHTML = `<div class="text-center text-error p-20 border border-outline-variant/30 rounded-xl bg-surface-container-lowest shadow-sm">Error generating answer: ${e}</div>`;
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<span class="material-symbols-outlined text-sm">auto_awesome</span> Ask AI`;
        }
    }

    init();
</script>
</body></html>"""

@app.get("/")
def serve_frontend():
    """Serve the single-page application HTML."""
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=7860, reload=True)
