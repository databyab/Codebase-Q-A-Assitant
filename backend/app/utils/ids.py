from __future__ import annotations

import hashlib
from urllib.parse import urlparse


def repo_id_from_url(repo_url: str) -> str:
    """Create a stable identifier from a repository URL."""
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/").removesuffix(".git")
    if not path:
        raise ValueError("Repository URL must include an owner and repository name.")
    normalized = path.replace("/", "__").lower()
    digest = hashlib.sha1(repo_url.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}__{digest}"


def repo_id_from_question(repo_id: str | None, repo_url: str | None) -> str:
    """Resolve a repo_id directly or derive it from the repository URL."""
    if repo_id:
        return repo_id
    if repo_url:
        return repo_id_from_url(repo_url)
    raise ValueError("Either repo_id or repo_url must be provided.")
