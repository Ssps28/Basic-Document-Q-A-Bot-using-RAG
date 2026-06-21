"""
ingest.py — Document ingestion pipeline for SmartDocs RAG.

Pipeline stages:
  1. Extract text from PDF/DOCX files in data/
  2. Split extracted text into overlapping chunks
  3. Embed each chunk using Gemini's embedding model (batched)
  4. Store chunks + embeddings in a persistent ChromaDB collection

Run directly with: python -m src.ingest
"""

import time
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument
import chromadb
from google import genai
from google.genai import types as genai_types

from src.config import (
    DATA_DIR,
    DB_DIR,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)

EMBED_BATCH_SIZE = 25  # chunks per API call — stays under Gemini's per-request limits


# ---------------------------------------------------------------------------
# Stage 1: Extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(file_path: Path) -> list[dict]:
    """Extracts text page-by-page from a PDF."""
    extracted = []
    file_name = file_path.name

    try:
        reader = PdfReader(str(file_path))
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1
                    }
                })
    except Exception as e:
        print(f"  ERROR reading PDF {file_name}: {e}")

    return extracted


def extract_docx_text(file_path: Path) -> list[dict]:
    """Extracts text from a DOCX file (no native page concept)."""
    extracted = []
    file_name = file_path.name

    try:
        doc = DocxDocument(str(file_path))
        full_text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        combined = " ".join(full_text)
        if combined.strip():
            extracted.append({
                "text": combined,
                "metadata": {
                    "source": file_name,
                    "page": 1
                }
            })
    except Exception as e:
        print(f"  ERROR reading DOCX {file_name}: {e}")

    return extracted


def load_all_documents() -> list[dict]:
    """Scans DATA_DIR and dispatches each file to the right extractor."""
    all_pages = []

    if not DATA_DIR.exists():
        print(f"ERROR: data directory not found at {DATA_DIR}")
        return all_pages

    files = sorted(DATA_DIR.glob("*"))
    if not files:
        print(f"WARNING: no files found in {DATA_DIR}")

    for file_path in files:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            print(f"Extracting PDF: {file_path.name}")
            all_pages.extend(extract_pdf_pages(file_path))
        elif suffix == ".docx":
            print(f"Extracting DOCX: {file_path.name}")
            all_pages.extend(extract_docx_text(file_path))
        else:
            print(f"Skipping unsupported file: {file_path.name}")

    return all_pages


# ---------------------------------------------------------------------------
# Stage 2: Chunking
# ---------------------------------------------------------------------------

def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits page-level text into smaller, overlapping chunks.
    Overlap ensures that information near a chunk boundary isn't lost.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        text_length = len(text)

        if text_length <= chunk_size:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": text,
                "metadata": {**metadata, "chunk_index": 0}
            })
            chunk_id += 1
            continue

        start = 0
        local_index = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "metadata": {**metadata, "chunk_index": local_index}
            })

            chunk_id += 1
            local_index += 1
            start += (chunk_size - chunk_overlap)

    return chunks


# ---------------------------------------------------------------------------
# Stage 3: Embedding (batched, with retry on rate limits)
# ---------------------------------------------------------------------------

def embed_chunks_in_batches(chunks: list[dict], client: genai.Client) -> list[list[float]]:
    """
    Embeds all chunk texts using the Gemini embedding model, in batches.
    Retries automatically on 429 (rate limit) errors.
    """
    all_embeddings = []
    texts = [c["text"] for c in chunks]
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        batch_num = (i // EMBED_BATCH_SIZE) + 1

        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                batch_vectors = [emb.values for emb in result.embeddings]
                all_embeddings.extend(batch_vectors)
                break  # success

            except Exception as e:
                error_text = str(e)
                is_rate_limit = "429" in error_text or "RESOURCE_EXHAUSTED" in error_text

                if is_rate_limit and attempt < max_retries:
                    wait_seconds = 35
                    print(f"    Rate limit hit (attempt {attempt}/{max_retries}). Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                else:
                    raise

        # Pacing delay between every batch to avoid tripping the free-tier quota
        time.sleep(3)

    return all_embeddings


# ---------------------------------------------------------------------------
# Stage 4: Vector database storage
# ---------------------------------------------------------------------------

def save_to_vector_db(chunks: list[dict], embeddings: list[list[float]]):
    """Stores chunks + their embeddings in a persistent local ChromaDB collection."""
    db_client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        db_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    collection = db_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"\nSuccessfully stored {len(chunks)} chunks in ChromaDB at: {DB_DIR}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_ingestion_pipeline():
    """Full pipeline: extract -> chunk -> embed -> store."""
    print("=== Step 1: Extracting documents ===")
    pages = load_all_documents()
    print(f"Extracted {len(pages)} page-chunks from source documents.\n")

    print("=== Step 2: Chunking text ===")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).\n")

    print("=== Step 3: Generating embeddings (batched) ===")
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    embeddings = embed_chunks_in_batches(chunks, genai_client)
    print(f"Generated {len(embeddings)} embedding vectors.\n")

    print("=== Step 4: Saving to vector database ===")
    save_to_vector_db(chunks, embeddings)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    run_ingestion_pipeline()