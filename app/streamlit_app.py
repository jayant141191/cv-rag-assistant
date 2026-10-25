"""Streamlit UI for the CV RAG Assistant.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_rag.config import get_settings  # noqa: E402
from cv_rag.graph import CVRagGraph  # noqa: E402
from cv_rag.loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from cv_rag.vectorstore import CVVectorStore  # noqa: E402

st.set_page_config(page_title="CV RAG Assistant", page_icon="📄", layout="wide")

settings = get_settings()

if not settings.openai_api_key:
    st.error("`OPENAI_API_KEY` is not set. Copy `.env.example` to `.env`, add your key, and restart.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_backend():
    store = CVVectorStore(settings)
    graph = CVRagGraph(retriever=lambda q, cands: store.search(q, cands), settings=settings)
    return store, graph


store, graph = load_backend()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources", "trace"}]

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📁 CV library")
    uploads = st.file_uploader(
        "Upload CVs", type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded CVs", type="primary"):
        progress = st.progress(0.0)
        for i, f in enumerate(uploads, start=1):
            try:
                info = store.add_cv(f.getvalue(), f.name)
                st.success(f"{info['candidate']} — {info['chunks']} chunks")
            except Exception as exc:  # show the error, keep going with other files
                st.error(f"{f.name}: {exc}")
            progress.progress(i / len(uploads))

    sources = store.list_sources()
    st.subheader(f"Indexed ({len(sources)})")
    if not sources:
        st.caption("No CVs yet. Upload some above, or run `python scripts/ingest.py data/sample_cvs`.")
    for src in sources:
        col1, col2 = st.columns([5, 1])
        col1.markdown(f"**{src['candidate']}**  \n<small>{src['source']} · {src['chunks']} chunks</small>",
                      unsafe_allow_html=True)
        if col2.button("✕", key=f"del-{src['source']}", help=f"Remove {src['source']}"):
            store.delete_source(src["source"])
            st.rerun()

    st.divider()
    candidates = store.list_candidates()
    candidate_filter = st.multiselect(
        "Restrict answers to", candidates,
        help="Leave empty to search all CVs. The assistant also detects names in your question.",
    )
    show_trace = st.toggle("Show pipeline trace", value=False)

    c1, c2 = st.columns(2)
    if c1.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if c2.button("Reset index"):
        store.reset()
        st.session_state.messages = []
        st.rerun()

    st.caption(f"LLM: `{settings.chat_model}` · Embeddings: `{settings.embedding_model}`")

# ---------------------------------------------------------------------- chat
st.title("📄 CV RAG Assistant")
st.caption("Ask about candidates' experience, skills, education — or compare them.")


def render_extras(msg: dict) -> None:
    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(f"**[{s['index']}] {s['candidate']}** — {s['section']} · `{s['source']}`")
                st.text(s["text"])
    if show_trace and msg.get("trace"):
        with st.expander("Pipeline trace"):
            for step in msg["trace"]:
                st.markdown(f"- {step}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_extras(msg)

if not st.session_state.messages and candidates:
    st.info("Try: *Who has the most experience with AWS?* · *Summarise "
            f"{candidates[0]}'s career* · *Compare the candidates' education*")

if question := st.chat_input("Ask a question about the CVs…"):
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not candidates:
            reply = {"role": "assistant", "content": "Please upload and index at least one CV first."}
            st.markdown(reply["content"])
        else:
            with st.spinner("Searching CVs…"):
                result = graph.ask(question, chat_history=history,
                                   known_candidates=candidates, candidate_filter=candidate_filter)
            reply = {"role": "assistant", "content": result["answer"],
                     "sources": result["sources"], "trace": result["trace"]}
            st.markdown(reply["content"])
            render_extras(reply)
    st.session_state.messages.append(reply)
