import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env at the project root
load_dotenv()

# --- API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Model Names ---
# gemini-2.5-flash: stable, GA, cost-efficient — good fit for RAG's
# "long input, short output" pattern (large context chunks in, short answer out)
GENERATION_MODEL = "gemini-2.5-flash"

# gemini-embedding-001: stable, GA, text-only embedding model
# (text-embedding-004 used in older tutorials has been shut down by Google)
EMBEDDING_MODEL = "gemini-embedding-001"

# --- Paths ---
# Path(__file__) = this config.py file's location
# .parent = src/ folder
# .parent again = project root (SmartDocs_RAG/)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

# --- Chunking Configuration ---
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between consecutive chunks

# --- Retrieval Configuration ---
TOP_K = 4                # number of chunks to retrieve per query

# --- ChromaDB Collection Name ---
COLLECTION_NAME = "finance_knowledge_base"