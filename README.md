# Codebase Q&A Assistant

This repository is split into two services:

- `backend/`: FastAPI + RAG pipeline
- `frontend/`: Streamlit UI

The backend ingests a GitHub repository, builds a persistent FAISS index over its code, and answers natural-language questions with source references using the Groq API.

## Architecture

1. Repository ingestion clones the target repo into `backend/data/raw_repos`, filters out binary and build artifacts, and chunks source files plus Jupyter notebooks with language-aware splitters.
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
│   ├── .env.example
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
├── .gitignore
├── docker-compose.yml
└── README.md
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

