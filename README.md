# Grounded RAG Chatbot over Financial Filings

A retrieval-augmented chatbot that answers questions **only** from a provided
set of financial documents — PDFs and Word reports — and explicitly refuses
to answer when the documents don't contain enough information, rather than
guessing. Built as a technical exercise in data engineering and RAG system
design.

**Live demo:** rag-chatbot-demo-7kbicza3x4dejhmn4adarh.streamlit.app

---

## What this is

Given documents from multiple departments/formats (PDF, Word, and — scoped
out below — Excel), the brief was to design a pipeline that can:

1. Ingest and structure heterogeneous document formats
2. Make the content searchable and groundable
3. Answer questions strictly from that content, with citations
4. Refuse cleanly when the answer isn't in the documents

## Dataset

| Format | Source | Why |
|---|---|---|
| **PDF** | 9 real SEC filings for 3M (2015–2022 10-Ks + Q2 2023 10-Q), sourced from the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark (Patronus AI) | Real, public, genuinely messy — multi-column layouts, large tables, 100–250+ pages per filing |
| **Word (.docx)** | 2 synthetic quarterly reports for a fictional company ("Solstice Retail Holdings") | No real public company files official quarterly results as `.docx` — SEC EDGAR only accepts PDF/HTML/XBRL, so no real equivalent exists to source. This is a deliberate choice, not a shortcut. |
| **Excel** | Out of scope for this build | Real supplemental Excel data exists for some companies (e.g. bank earnings supplements), but was deprioritized in favor of getting PDF and Word fully correct first, consistent with the stated approach going in |

Using FinanceBench also means the 8 real evaluation questions below — with
verified gold answers and source page citations — were **not written for
this project**. They're an independent, published benchmark, which makes
the evaluation results below a genuine test rather than a self-graded one.

## Architecture

```
Documents (PDF/DOCX)
   │
   ▼
Docling  ──────────────  structure-aware parsing: tables kept as tables,
                          not flattened into unreadable number runs
   │
   ▼
Chunking  ─────────────  character-window chunking with overlap; markdown
                          tables kept whole rather than split mid-row;
                          every chunk tagged with doc_name + page
   │
   ▼
Embedding  ────────────  all-MiniLM-L6-v2 (local, free, no API cost)
   │
   ▼
Chroma vector store  ──  stores vectors + metadata together
   │
   ▼
Retrieval  ────────────  top-k similarity search + a RELEVANCE FLOOR:
                          if the best match isn't close enough, the LLM
                          is never even called — this is the actual
                          mechanism behind "only answer from documents,"
                          not just a prompt instruction
   │
   ▼
Generation  ───────────  Claude (Anthropic API), system prompt enforces
                          citation + refusal behavior as a second,
                          behavioral layer on top of the structural floor
   │
   ▼
Streamlit UI  ─────────  this app
```

**Why a relevance floor and not just a system prompt:** prompt instructions
alone get ignored under pressure (models will often answer from weak context
rather than admit uncertainty). Gating retrieval with a numeric similarity
threshold means the model never sees irrelevant context in the first place —
structural grounding, not just behavioral.

## Evaluation results

Run against the real FinanceBench ground-truth set (8 questions, independently
verified answers and source pages):

- **Retrieval doc-match rate: 88% (7/8)** — correctly retrieved the right
  source filing for 7 of 8 questions
- **Zero hallucinations** — every refusal came with an honest, specific
  explanation of what was missing, never a fabricated number
- **Known limitation:** despite retrieving the correct *filing* in 6 of 7
  refused cases, the model couldn't always find the correct *page* within
  it — each filing runs 100–250+ pages, and the specific balance-sheet or
  cash-flow page competes against hundreds of other chunks from the same
  document. Raising `TOP_K` from 5 to 10 was tested as a cheap fix; it
  didn't meaningfully move the result, which tells us the real fix is
  section-aware chunking (tagging chunks by statement type during parsing)
  or hybrid keyword+embedding search, not just retrieving more chunks.

Separately, a synthetic Word-doc eval (2 real-answer questions + 1 refusal
trap) and a 4-question refusal sanity check (competitor data, stock
predictions, out-of-scope companies) both pass cleanly.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires a pre-built `chroma_store/` directory in the same folder (see
the ingestion notebook) and an `ANTHROPIC_API_KEY` set via Streamlit
secrets (`.streamlit/secrets.toml` locally, or the app dashboard when
deployed).

## Tech stack

Python · Docling · Sentence-Transformers · ChromaDB · Anthropic API
(Claude Sonnet) · Streamlit
