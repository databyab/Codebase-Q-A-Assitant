from __future__ import annotations

import json
from functools import lru_cache
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    HealthResponse,
    IngestRepoRequest,
    IngestRepoResponse,
    QuestionRequest,
    QuestionResponse,
)
from app.services.qa_service import QAService
from app.services.repository_service import RepositoryService

router = APIRouter(prefix="/api/v1", tags=["codebase-qa"])


@lru_cache(maxsize=1)
def get_repository_service() -> RepositoryService:
    """Return a lazily initialized repository service."""
    return RepositoryService()


@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    """Return a lazily initialized QA service."""
    return QAService()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Return a simple health payload for uptime checks."""
    return HealthResponse(status="ok")


@router.post(
    "/repos/ingest",
    response_model=IngestRepoResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_repository(payload: IngestRepoRequest) -> IngestRepoResponse:
    """Clone and index a repository for downstream question answering."""
    try:
        return get_repository_service().ingest_repository(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/qa/ask", response_model=QuestionResponse)
async def ask_question(payload: QuestionRequest) -> QuestionResponse:
    """Answer a natural language question against an indexed repository."""
    try:
        return await get_qa_service().answer_question(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/qa/ask/stream")
async def ask_question_stream(payload: QuestionRequest) -> StreamingResponse:
    """Stream an answer using Server-Sent Events."""
    try:
        stream, sources, repo_id = await get_qa_service().stream_answer(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    async def event_stream() -> AsyncIterator[str]:
        async for delta in stream:
            yield f"event: answer_delta\ndata: {json.dumps({'content': delta})}\n\n"
        final_payload = {
            "repo_id": repo_id,
            "sources": [source.model_dump() for source in sources],
        }
        yield f"event: done\ndata: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
