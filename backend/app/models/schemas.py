from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str


class IngestRepoRequest(BaseModel):
    """Request body for repository ingestion."""

    repo_url: str = Field(..., description="GitHub repository URL")
    branch: str | None = Field(default=None, description="Optional branch name")
    force_refresh: bool = Field(default=False)


class IngestRepoResponse(BaseModel):
    """Repository indexing result payload."""

    repo_id: str
    repo_url: str
    branch: str | None = None
    local_path: str
    index_path: str
    indexed_at: str
    total_chunks: int
    total_files: int
    skipped_files: int


class QuestionRequest(BaseModel):
    """Request body for question answering."""

    question: str
    repo_id: str | None = None
    repo_url: str | None = None
    top_k: int | None = Field(default=8, ge=1, le=20)
    fetch_k: int | None = Field(default=24, ge=4, le=60)
    max_context_chars: int | None = Field(default=18000, ge=4000, le=40000)

    @model_validator(mode="after")
    def validate_repo_identifier(self) -> "QuestionRequest":
        """Require either repo_id or repo_url."""
        if not self.repo_id and not self.repo_url:
            raise ValueError("Either repo_id or repo_url must be provided.")
        return self


class SourceReference(BaseModel):
    """Source citation returned alongside the answer."""

    file_path: str
    symbol_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    score: float | None = None


class QuestionResponse(BaseModel):
    """Standard question-answering response."""

    repo_id: str
    question: str
    answer: str
    sources: list[SourceReference]
