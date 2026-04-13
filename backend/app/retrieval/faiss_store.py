from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import get_settings


class FAISSVectorStore:
    """Manage persistent FAISS indexes on local disk."""

    def __init__(self, embeddings: Embeddings) -> None:
        self.settings = get_settings()
        self.embeddings = embeddings

    def build_and_persist(
        self,
        repo_id: str,
        documents: list[Document],
        manifest: dict[str, Any],
    ) -> None:
        """Create and persist a FAISS index plus a manifest."""
        target_dir = self.settings.vector_store_dir / repo_id
        target_dir.mkdir(parents=True, exist_ok=True)
        store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        store.save_local(str(target_dir))
        (target_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def load(self, repo_id: str) -> tuple[FAISS, dict[str, Any]]:
        """Load a persisted FAISS index and metadata manifest."""
        target_dir = self.settings.vector_store_dir / repo_id
        if not target_dir.exists():
            raise FileNotFoundError(
                f"No index found for repo_id '{repo_id}'. Ingest the repository first."
            )

        manifest_path = target_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Index manifest missing for repo_id '{repo_id}'.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = FAISS.load_local(
            str(target_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return store, manifest

    def manifest_path(self, repo_id: str) -> Path:
        """Return the location of the repo manifest."""
        return self.settings.vector_store_dir / repo_id / "manifest.json"
