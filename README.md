\# 💰 SmartDocs RAG — Personal Finance Document Q\&A



A Retrieval-Augmented Generation (RAG) app that answers personal finance

questions — budgeting, saving, credit, retirement, and taxes — using only

a curated set of official finance PDFs as its knowledge source. Built with

Gemini, ChromaDB, and Streamlit.



\## How it works



1\. \*\*Ingest\*\* — PDFs in `data/` are chunked and embedded with Gemini's

&#x20;  `gemini-embedding-001` model, then stored in a local ChromaDB collection.

2\. \*\*Query\*\* — A user's question is embedded and matched against the

&#x20;  collection to retrieve the top-k most relevant chunks.

3\. \*\*Generate\*\* — Those chunks are passed to `gemini-2.5-flash` as grounded

&#x20;  context, and the model answers using \*only\* that context, with inline

&#x20;  citations like `(source\_filename, Page X)`.



\## Project structure



\## Setup



\*\*1. Create and activate a virtual environment\*\*

```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```

> If activation is blocked by execution policy, run once:

> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`



\*\*2. Install dependencies\*\*

```powershell

pip install -r requirements.txt

```



\*\*3. Add your Gemini API key\*\*



Create a `.env` file in the project root:



Get a key from \[Google AI Studio](https://aistudio.google.com/apikey).

> Note: some accounts currently get issued a non-standard `AQ.`-prefixed

> key, which fails with `401 ACCESS\_TOKEN\_TYPE\_UNSUPPORTED`. If that

> happens, try generating the key under a different/older Google Cloud

> project, or use Vertex AI auth instead of an API key.



\*\*4. Build the vector database\*\*

```powershell

python -m src.ingest

```

This reads everything in `data/`, chunks it, embeds it, and writes the

result to `db/`. Re-run this any time you add or change PDFs.



\## Usage



\*\*Web UI\*\*

```powershell

streamlit run streamlit\_app.py

```

Then open `http://localhost:8501`.



\*\*Command line\*\*

```powershell

python -m src.main

```



\## Configuration



Adjust retrieval/chunking behavior in `src/config.py`:



| Setting | Default | Description |

|---|---|---|

| `CHUNK\_SIZE` | 1000 | Characters per chunk |

| `CHUNK\_OVERLAP` | 200 | Overlap between chunks |

| `TOP\_K` | 4 | Chunks retrieved per query |

| `GENERATION\_MODEL` | `gemini-2.5-flash` | Answer generation |

| `EMBEDDING\_MODEL` | `gemini-embedding-001` | Embeddings |



\## Notes



\- If the question can't be answered from the provided documents, the

&#x20; model responds with: \*"I cannot find the answer to this in the

&#x20; provided documents."\* — it won't fall back on outside knowledge.

\- `venv/`, `db/`, and `.env` are gitignored; clone + follow Setup above

&#x20; to reproduce the environment from scratch.





