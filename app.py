"""
Grounded RAG chatbot — Streamlit app.

Serves the same pipeline proven in the Kaggle notebook (retrieval with a
relevance floor, citation-forced generation, refusal when nothing relevant
is found) as a live, clickable chat interface.

Expects a pre-built Chroma vector store at ./chroma_store (download this
from your Kaggle notebook's Output tab and place it next to this file
before deploying — see the deploy steps).
"""

import re
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# ---------------------------------------------------------------------------
# Config — same constants as the notebook
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10
RELEVANCE_FLOOR = 0.35
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "silverline_filings"
MODEL_NAME = "claude-sonnet-5-5"

# Rough per-token pricing for claude-sonnet-5-5, USD per token — for a
# running cost estimate shown in the UI, not for billing.
INPUT_COST_PER_TOKEN = 3.00 / 1_000_000
OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000

SYSTEM_PROMPT = """You are a financial-documents assistant. You answer ONLY using the
context provided below, which is extracted from the company's filings. Rules:
- If the context does not contain enough information to answer, say exactly:
  "I don't have enough information in the provided documents to answer that."
- Never use outside knowledge about the company, the industry, or anything else.
- When you do answer, cite the source filing and page number(s) you used.
- Be precise with numbers — do not round or approximate figures from the context.
- The question may be phrased differently than the source text (e.g. "did X
  happen" instead of the document's own wording). Synthesize the answer from
  what the context actually supports, even when the wording doesn't match
  literally — don't refuse just because the question uses different words
  than the document, if the context answers it.
"""

st.set_page_config(page_title="Financial Filings RAG Chatbot", page_icon="📊")


# ---------------------------------------------------------------------------
# PII guardrail — mask Aadhaar/PAN-shaped numbers before anything reaches
# Claude's API (i.e. before it leaves our infra), unmask in the response.
# Requested on the Nov 2 call: don't send Aadhaar/PAN/account numbers to a
# US-hosted LLM even if the user types them into the question.
# ---------------------------------------------------------------------------
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")


def mask_pii(text: str) -> tuple[str, dict[str, str]]:
    """Replace Aadhaar/PAN-shaped numbers with placeholders. Returns the
    masked text and a map to restore the originals afterward."""
    redactions: dict[str, str] = {}
    counter = 0

    def _replace(pattern_name):
        def _sub(m):
            nonlocal counter
            counter += 1
            placeholder = f"[REDACTED_{pattern_name}_{counter}]"
            redactions[placeholder] = m.group(0)
            return placeholder
        return _sub

    masked = AADHAAR_RE.sub(_replace("AADHAAR"), text)
    masked = PAN_RE.sub(_replace("PAN"), masked)
    return masked, redactions


def unmask_pii(text: str, redactions: dict[str, str]) -> str:
    for placeholder, original in redactions.items():
        text = text.replace(placeholder, original)
    return text


# ---------------------------------------------------------------------------
# Cached resources — loaded once per session, not on every message
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to vector store...")
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def load_claude_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "No ANTHROPIC_API_KEY found in Streamlit Secrets. "
            "Add it under App settings → Secrets before using the chat."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


embedder = load_embedder()
collection = load_collection()
claude = load_claude_client()


# ---------------------------------------------------------------------------
# Pipeline — identical logic to the notebook
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K):
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 / (1 + dist)
        hits.append(
            {
                "text": doc,
                "doc_name": meta["doc_name"],
                "page": meta["page"],
                "score": similarity,
            }
        )
    return hits


def is_grounded_enough(hits: list[dict]) -> bool:
    return bool(hits) and hits[0]["score"] >= RELEVANCE_FLOOR


def answer_question(question: str) -> dict:
    hits = retrieve(question)

    if not is_grounded_enough(hits):
        return {
            "answer": "I don't have enough information in the provided documents to answer that.",
            "sources": [],
            "refused": True,
        }

    context = "\n\n---\n\n".join(
        f"[Source: {h['doc_name']}, page {h['page']}]\n{h['text']}" for h in hits
    )

    # Mask any Aadhaar/PAN-shaped numbers in the question before it leaves
    # our infra. The retrieved context comes from our own documents, not
    # user input, so it isn't masked — only what the user typed is at risk.
    masked_question, redactions = mask_pii(question)

    response = claude.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {masked_question}"}
        ],
    )

    # Defensive extraction — response.content can include a thinking block
    # before the text block, or (rarely) no text block at all if generation
    # is cut off or filtered. A bare next(...) on an empty generator throws
    # StopIteration and crashes the whole app — this happened live on Nov 2.
    answer_text = next(
        (block.text for block in response.content if block.type == "text"),
        None,
    )
    if answer_text is None:
        answer_text = "I wasn't able to generate a response to that — please rephrase or try again."

    # Stitch any redacted PII back in, in case the model echoed a placeholder.
    answer_text = unmask_pii(answer_text, redactions)

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    call_cost = input_tokens * INPUT_COST_PER_TOKEN + output_tokens * OUTPUT_COST_PER_TOKEN
    st.session_state.setdefault("usage_log", []).append(
        {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": call_cost}
    )

    return {
        "answer": answer_text,
        "sources": [
            {"doc_name": h["doc_name"], "page": h["page"], "score": round(h["score"], 3)}
            for h in hits
        ],
        "refused": False,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 Financial Filings RAG Chatbot")
st.caption(
    "Grounded over Silverline Industries' real financial filings — the "
    "auditor's limited review report, board meeting minutes, and financial "
    "results/resolutions spreadsheets. Answers only from the documents — "
    "refuses when nothing relevant is found, rather than guessing."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

usage_log = st.session_state.setdefault("usage_log", [])
with st.sidebar:
    st.subheader("Session usage")
    if usage_log:
        total_in = sum(u["input_tokens"] for u in usage_log)
        total_out = sum(u["output_tokens"] for u in usage_log)
        total_cost = sum(u["cost_usd"] for u in usage_log)
        st.write(f"Calls: {len(usage_log)}")
        st.write(f"Input tokens: {total_in}")
        st.write(f"Output tokens: {total_out}")
        st.write(f"Estimated cost: ${total_cost:.4f}")
    else:
        st.write("No generation calls yet this session.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s['doc_name']}, page {s['page']} (score: {s['score']})")

if question := st.chat_input("Ask about the filings..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.write(f"- {s['doc_name']}, page {s['page']} (score: {s['score']})")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
