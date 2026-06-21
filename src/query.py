"""
query.py — Retrieval + answer generation for SmartDocs RAG.

Given a user question:
  1. Embed the question (using RETRIEVAL_QUERY task type)
  2. Search ChromaDB for the top-k most similar chunks
  3. Build a grounded prompt with those chunks as context
  4. Ask Gemini to answer using ONLY that context, with citations
"""

import chromadb
from google import genai
from google.genai import types as genai_types

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    DB_DIR,
    COLLECTION_NAME,
    TOP_K,
)

SYSTEM_PROMPT = (
    "You are a precise document Q&A assistant for a personal finance knowledge base. "
    "Answer the user's question using ONLY the provided context below — never use "
    "your own outside knowledge, even if you know the answer. "
    "If the answer cannot be found in the context, respond exactly with: "
    "\"I cannot find the answer to this in the provided documents.\" "
    "When you do answer, cite your sources inline using the format "
    "(source_filename, Page X) right after the relevant fact."
)


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def get_collection():
    """Loads the existing persistent ChromaDB collection (does NOT re-embed anything)."""
    db_client = chromadb.PersistentClient(path=str(DB_DIR))
    return db_client.get_collection(name=COLLECTION_NAME)


def retrieve_relevant_chunks(question: str, client: genai.Client, collection, top_k: int = TOP_K) -> dict:
    """
    Embeds the user's question and retrieves the top_k most similar chunks
    from ChromaDB, along with their metadata and similarity distance.
    """
    embed_result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[question],
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = embed_result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )
    return results


def build_grounded_prompt(question: str, results: dict) -> tuple[str, list[str]]:
    """
    Formats retrieved chunks into a labeled context block, and builds
    the final prompt sent to Gemini. Also returns a clean citation list
    for display purposes (separate from what's embedded in the answer text).
    """
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_blocks = []
    citations = []

    for doc_text, meta, distance in zip(documents, metadatas, distances):
        source = meta["source"]
        page = meta["page"]
        citation_str = f"{source}, Page {page}"

        context_blocks.append(f"[Source: {citation_str}]\n{doc_text}")
        citations.append(citation_str)

    context_payload = "\n\n---\n\n".join(context_blocks)

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT FROM DOCUMENTS:\n{context_payload}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    return full_prompt, citations


def generate_answer(prompt: str, client: genai.Client) -> str:
    """Sends the grounded prompt to Gemini and returns the generated answer text."""
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text


def answer_question(question: str, client: genai.Client = None, collection=None, top_k: int = TOP_K) -> dict:
    """
    Full query pipeline, callable from main.py or streamlit_app.py.
    Returns a dict with the answer text, citation list, and raw retrieved chunks
    (the raw chunks let the UI show "what context was used" per the assignment's
    requirement to display retrieved source chunks alongside the answer).
    """
    if client is None:
        client = get_genai_client()
    if collection is None:
        collection = get_collection()

    results = retrieve_relevant_chunks(question, client, collection, top_k)
    prompt, citations = build_grounded_prompt(question, results)
    answer = generate_answer(prompt, client)

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": list(zip(results["documents"][0], results["metadatas"][0])),
    }


# --- Quick manual test when running this file directly ---
if __name__ == "__main__":
    test_question = "Why is having a budget important?"
    print(f"Question: {test_question}\n")

    result = answer_question(test_question)

    print("Answer:")
    print(result["answer"])
    print("\nSources used:")
    for c in result["citations"]:
        print(f"  - {c}")