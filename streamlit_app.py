"""
streamlit_app.py — Web UI for SmartDocs RAG.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st

from src.query import get_genai_client, get_collection, answer_question
from src.config import TOP_K

st.set_page_config(page_title="SmartDocs RAG — Finance Q&A", page_icon="💰")


@st.cache_resource
def load_resources():
    """
    Loads the Gemini client and ChromaDB collection ONCE per app session,
    not on every question. @st.cache_resource is the correct tool here
    because we're caching shared objects (a client, a DB connection),
    not data — Streamlit reruns this whole script on every interaction,
    so without caching we'd reconnect to the DB on every single click.
    """
    client = get_genai_client()
    collection = get_collection()
    return client, collection


st.title("💰 SmartDocs RAG")
st.caption("Ask questions about budgeting, saving, credit, retirement, and taxes — answered from a curated set of official finance documents.")

try:
    client, collection = load_resources()
except Exception as e:
    st.error(
        "Could not load the vector database. "
        "Did you run `python -m src.ingest` first?"
    )
    st.exception(e)
    st.stop()

question = st.text_input("Your question:", placeholder="e.g. Why is having a budget important?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Searching documents and generating answer..."):
        result = answer_question(question, client=client, collection=collection, top_k=TOP_K)

    st.markdown("### Answer")
    st.markdown(result["answer"])

    st.markdown("### Sources Used")
    for citation in result["citations"]:
        st.markdown(f"- {citation}")

    with st.expander("View retrieved source chunks (raw context sent to the model)"):
        for i, (chunk_text, meta) in enumerate(result["retrieved_chunks"], start=1):
            st.markdown(f"**Chunk {i}** — *{meta['source']}, Page {meta['page']}*")
            st.text(chunk_text)
            st.divider()

elif ask_clicked and not question.strip():
    st.warning("Please type a question first.")