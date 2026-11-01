"""
Grounded RAG chatbot — Streamlit app.

Serves the same pipeline proven in the Kaggle notebook (retrieval with a
relevance floor, citation-forced generation, refusal when nothing relevant
is found) as a live, clickable chat interface.

Expects a pre-built Chroma vector store at ./chroma_store (download this
from your Kaggle notebook's Output tab and place it next to this file
before deploying — see the deploy steps).
"""

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
COLLECTION_NAME = "3m_filings"
MODEL_NAME = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are a financial-documents assistant. You answer ONLY using the
context provided below, which is extracted from the company's SEC filings and quarterly
reports. Rules:
- If the context does not contain enough information to answer, say exactly:
  "I don't have enough information in the provided documents to answer that."
- Never use outside knowledge about the company, the industry, or anything else.
- When you do answer, cite the source filing and page number(s) you used.
- Be precise with numbers — do not round or approximate figures from the context.
"""

st.set_page_config(page_title="Financial Filings RAG Chatbot", page_icon="📊")


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

    response = claude.messages.create(
        model=MODEL_NAME,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
    )

    answer_text = next(
        block.text for block in response.content if block.type == "text"
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
    "Grounded over real 3M SEC filings (2015–2022 10-Ks, Q2 2023 10-Q) and "
    "sample quarterly reports. Answers only from the documents — refuses "
    "when nothing relevant is found, rather than guessing."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
