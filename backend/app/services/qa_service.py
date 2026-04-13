from __future__ import annotations

import logging
from typing import AsyncIterator

from langchain_core.documents import Document

from app.core.config import get_settings
from app.embeddings.huggingface import SentenceTransformerEmbeddings
from app.llm.groq_client import GroqLLMClient
from app.models.schemas import QuestionRequest, QuestionResponse, SourceReference
from app.retrieval.faiss_store import FAISSVectorStore
from app.utils.ids import repo_id_from_question

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior backend engineer helping a developer understand a codebase.
Answer only from the provided repository context.
If the answer is not fully supported by the retrieved code, say what is missing.
Be concrete, reference important files, and explain control flow when relevant.
Prefer concise, technically accurate explanations over speculation."""


class QAService:
    """Answer natural language questions over an indexed codebase."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=self.settings.embedding_model_name
        )
        self.vector_store = FAISSVectorStore(self.embeddings)
        self._llm_client: GroqLLMClient | None = None

    async def answer_question(self, payload: QuestionRequest) -> QuestionResponse:
        """Return a non-streaming answer with source references."""
        repo_id, context, sources = self._prepare_context(payload)
        answer = await self.llm_client.generate_answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(payload.question, context),
        )
        return QuestionResponse(
            repo_id=repo_id,
            question=payload.question,
            answer=answer,
            sources=sources,
        )

    async def stream_answer(
        self,
        payload: QuestionRequest,
    ) -> tuple[AsyncIterator[str], list[SourceReference], str]:
        """Return an async text stream and final source list."""
        repo_id, context, sources = self._prepare_context(payload)
        stream = self.llm_client.stream_answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(payload.question, context),
        )
        return stream, sources, repo_id

    @property
    def llm_client(self) -> GroqLLMClient:
        """Create the Groq client lazily so indexing can work without an API key."""
        if self._llm_client is None:
            self._llm_client = GroqLLMClient()
        return self._llm_client

    def _prepare_context(
        self,
        payload: QuestionRequest,
    ) -> tuple[str, str, list[SourceReference]]:
        repo_id = repo_id_from_question(payload.repo_id, payload.repo_url)
        store, _manifest = self.vector_store.load(repo_id=repo_id)
        documents_and_scores = store.similarity_search_with_score(
            query=payload.question,
            k=payload.fetch_k or self.settings.retrieval_fetch_k,
        )
        if not documents_and_scores:
            raise ValueError("No indexed context was retrieved for this question.")
        context, sources = self._build_context(documents_and_scores, payload)
        return repo_id, context, sources

    def _build_context(
        self,
        documents_and_scores: list[tuple[Document, float]],
        payload: QuestionRequest,
    ) -> tuple[str, list[SourceReference]]:
        """Select diverse chunks under a context budget."""
        max_chars = payload.max_context_chars or self.settings.retrieval_max_context_chars
        top_k = payload.top_k or self.settings.retrieval_top_k
        seen_sources: set[tuple[str, int | None, int | None]] = set()
        used_chars = 0
        context_blocks: list[str] = []
        sources: list[SourceReference] = []

        for document, score in documents_and_scores:
            metadata = document.metadata
            file_path = metadata.get("file_path", "unknown")
            line_key = (
                file_path,
                metadata.get("start_line"),
                metadata.get("end_line"),
            )
            if line_key in seen_sources:
                continue

            chunk_text = document.page_content.strip()
            if not chunk_text:
                continue

            reserve = len(chunk_text) + 250
            if used_chars + reserve > max_chars:
                continue

            used_chars += reserve
            seen_sources.add(line_key)
            context_blocks.append(
                "\n".join(
                    [
                        f"FILE: {file_path}",
                        f"SYMBOL: {metadata.get('symbol_name', 'N/A')}",
                        f"LINES: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}",
                        "CODE:",
                        chunk_text,
                    ]
                )
            )
            sources.append(
                SourceReference(
                    file_path=file_path,
                    symbol_name=metadata.get("symbol_name"),
                    start_line=metadata.get("start_line"),
                    end_line=metadata.get("end_line"),
                    score=round(float(score), 4),
                )
            )
            if len(sources) >= top_k:
                break

        if not context_blocks:
            raise ValueError("Relevant chunks were retrieved, but none fit the context budget.")

        logger.info(
            "Built context blocks=%s sources=%s max_chars=%s",
            len(context_blocks),
            len(sources),
            max_chars,
        )
        return "\n\n---\n\n".join(context_blocks), sources

    @staticmethod
    def _build_user_prompt(question: str, context: str) -> str:
        """Format the final question with retrieved repository context."""
        return f"""Repository context:
{context}

Question:
{question}

Instructions:
- Explain the answer with concrete references to the retrieved files.
- Mention uncertainties if the code shown is incomplete.
- End with a short 'Relevant files' line listing the most useful file paths."""
