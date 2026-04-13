# Codebase Q&A Assistant

This repository is split into two services:

- `backend/`: FastAPI + RAG pipeline
- `frontend/`: Streamlit UI

The backend ingests a GitHub repository, builds a persistent FAISS index over its code, and answers natural-language questions with source references using the Groq API.

## Architecture

1. Repository ingestion clones the target repo into `data/raw_repos`, filters out binary and build artifacts, and chunks source files plus Jupyter notebooks with language-aware splitters.
2. The embedding layer uses `sentence-transformers` to generate local embeddings, then stores vectors in a persistent per-repository FAISS index.
3. The QA path retrieves relevant chunks, constrains context to a fixed budget, and sends only the best code snippets to Groq for low-latency answer generation.
4. Every chunk preserves metadata such as file path, symbol name, and line ranges so answers can cite their sources.

## Folder Structure

```text
repo-root/
├── backend/
│   ├── app/
│   ├── data/
│   ├── scripts/
│   ├── .env
│   ├── Dockerfile
│   ├── README.md
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```

## Running Locally

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## API Examples

### Ingest a repository

```bash
curl -X POST http://localhost:8000/api/v1/repos/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"repo_url\":\"https://github.com/tiangolo/fastapi\"}"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/qa/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"repo_url\":\"https://github.com/tiangolo/fastapi\",\"question\":\"Where is authentication handled?\"}"
```

## Design Decisions

- Per-repository indexes keep ingestion isolated and make it easy to cache, delete, or re-index specific repositories.
- AST-aware chunking for Python preserves function and class boundaries, which improves retrieval quality for code explanation prompts.
- Notebook support flattens `.ipynb` cells into retrieval-ready text blocks so data science and ML repositories can be indexed too.
- Context assembly uses a hard character budget to prevent overfilling the LLM context window on large repositories.
- Streaming is exposed through Server-Sent Events so a minimal frontend can render partial tokens quickly.

## Scalability Notes

- Move ingestion to a background worker queue for large repositories or multi-user workloads.
- Add commit-hash-based incremental indexing to skip unchanged files.
- Layer in reranking or graph-based retrieval for monorepos with many similar files.
- Keep hot indexes in memory and cold indexes on durable storage if traffic grows.

## Swapping FAISS to Pinecone

The vector store is isolated in `app/retrieval/faiss_store.py`. Replacing FAISS with Pinecone only requires swapping that adapter and persisting repo manifests in a remote store or database. The chunking, embedding, and QA services can remain unchanged.

## Deployment Notes

- Run the FastAPI app behind Uvicorn or Gunicorn with ASGI workers.
- Mount persistent storage for `data/raw_repos` and `data/vector_store`.
- Inject `GROQ_API_KEY` through the deployment environment.
- Add request metrics, ingestion duration metrics, and tracing in production.

## Docker Compose

From the repo root:

```bash
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`
