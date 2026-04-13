from __future__ import annotations

from typing import Sequence

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible sentence-transformers embedding wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        vectors = self.model.encode(
            list(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        vector = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vector.tolist()
