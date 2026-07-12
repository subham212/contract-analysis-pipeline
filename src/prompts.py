"""
prompts.py
==========
Prompt templates for clause extraction and contract summarization.

We support two modes:
  - zero_shot:   the model is given just the contract and instructions.
  - few_shot:    the model is given 2 worked examples before the real task.

All prompts return STRICT JSON so downstream parsing is deterministic.
"""
from __future__ import annotations

from textwrap import dedent

# ─── System prompt: sets the persona for every call ──────────────
SYSTEM_PROMPT = dedent("""\
    You are a meticulous legal contract analyst.
    You read contract text carefully and extract exact clause language.
    You NEVER invent or hallucinate text that is not in the contract.
    If a clause type is not present in the text, you say "Not found".
    You always answer in the exact requested format.
""")

# ─── Clause extraction: zero-shot ────────────────────────────────
EXTRACT_CLAUSES_ZERO_SHOT = dedent("""\
    Below is the full text of a legal contract (or a section of one).
    Your job: find and return the exact text of three clause types.

    CLAUSE TYPES TO FIND:
    1. termination       — conditions under which the agreement can be ended,
                           cancelled, or terminated by either party.
    2. confidentiality   — obligations to protect proprietary or secret
                           information.
    3. liability         — limitations of liability, indemnification,
                           caps on damages, or who bears risk.

    RULES:
    - Quote the ACTUAL text from the contract (do not paraphrase).
    - If a clause type is not present in this excerpt, write "Not found".
    - Keep each clause to its most relevant paragraph(s) (max ~500 words).

    CONTRACT TEXT:
    <<<
    {contract_text}
    >>>

    Respond ONLY with this exact JSON shape and nothing else:
    {{
      "termination": "<exact clause text or 'Not found'>",
      "confidentiality": "<exact clause text or 'Not found'>",
      "liability": "<exact clause text or 'Not found'>"
    }}
""")

# ─── Clause extraction: few-shot ─────────────────────────────────
# Two compact worked examples teach the model the expected output shape.
EXTRACT_CLAUSES_FEW_SHOT = dedent("""\
    You extract three clause types from contracts: termination,
    confidentiality, and liability. You quote the exact text and return JSON.

    Here are two examples of the expected behaviour.

    ── EXAMPLE 1 ──────────────────────────────────────────────────
    CONTRACT EXCERPT:
    "8. TERMINATION. Either party may terminate this Agreement upon thirty
    (30) days written notice. 12. CONFIDENTIALITY. Each party shall keep
    confidential all proprietary information. 15. The Provider's total
    liability shall not exceed the fees paid in the prior twelve months."

    ANSWER:
    {{
      "termination": "Either party may terminate this Agreement upon thirty (30) days written notice.",
      "confidentiality": "Each party shall keep confidential all proprietary information.",
      "liability": "The Provider's total liability shall not exceed the fees paid in the prior twelve months."
    }}

    ── EXAMPLE 2 ──────────────────────────────────────────────────
    CONTRACT EXCERPT:
    "Article 4. PAYMENT. Client shall pay $10,000 monthly. Article 9.
    NON-DISCLOSURE. The Receiving Party agrees not to disclose any
    Confidential Information for a period of three (3) years."

    ANSWER:
    {{
      "termination": "Not found",
      "confidentiality": "The Receiving Party agrees not to disclose any Confidential Information for a period of three (3) years.",
      "liability": "Not found"
    }}
    ───────────────────────────────────────────────────────────────

    Now do the same for the contract below.

    CONTRACT TEXT:
    <<<
    {contract_text}
    >>>

    Respond ONLY with this exact JSON shape:
    {{
      "termination": "<exact clause text or 'Not found'>",
      "confidentiality": "<exact clause text or 'Not found'>",
      "liability": "<exact clause text or 'Not found'>"
    }}
""")

# ─── Summarization prompt ────────────────────────────────────────
SUMMARIZE_CONTRACT = dedent("""\
    Below is the text of a legal contract.
    Write a concise summary of 100 to 150 words that captures:

    1. PURPOSE     — what the agreement is for (the deal).
    2. OBLIGATIONS — the key duties of each party.
    3. RISKS       — notable risks, penalties, or exposures.

    RULES:
    - Be factual. Do not add information not in the contract.
    - Keep it between 100 and 150 words (be precise).
    - Plain professional English, no markdown, no headings.

    CONTRACT TEXT:
    <<<
    {contract_text}
    >>>

    Respond with ONLY the summary paragraph, nothing else.
""")


def get_extraction_prompt(contract_text: str, mode: str = "zero_shot") -> str:
    """Return the right extraction prompt for the selected mode."""
    if mode == "few_shot":
        return EXTRACT_CLAUSES_FEW_SHOT.format(contract_text=contract_text)
    return EXTRACT_CLAUSES_ZERO_SHOT.format(contract_text=contract_text)


def get_summary_prompt(contract_text: str) -> str:
    """Return the summarization prompt."""
    return SUMMARIZE_CONTRACT.format(contract_text=contract_text)
