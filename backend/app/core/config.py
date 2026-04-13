from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "Codebase Q&A Assistant"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model_name: str = "openai/gpt-oss-120b"
    groq_temperature: float = 0.1
    groq_max_completion_tokens: int = 1400

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 8
    retrieval_fetch_k: int = 24
    retrieval_max_context_chars: int = 18000

    chunk_size: int = 1200
    chunk_overlap: int = 180
    max_file_size_bytes: int = 1_500_000

    data_dir: Path = BACKEND_ROOT / "data"
    raw_repos_dir: Path = data_dir / "raw_repos"
    vector_store_dir: Path = data_dir / "vector_store"

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton settings object."""
    settings = Settings()
    settings.raw_repos_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    return settings
