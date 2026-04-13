from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(slots=True)
class NormalizedRepoTarget:
    """Normalized repository target derived from a GitHub URL."""

    repo_url: str
    branch: str | None = None


def normalize_github_repo_target(
    repo_url: str,
    branch: str | None = None,
) -> NormalizedRepoTarget:
    """Normalize GitHub repo, tree, and blob URLs into a cloneable repo URL.

    Supported inputs:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo/tree/<branch>
    - https://github.com/owner/repo/blob/<branch>/<path>
    """
    parsed = urlparse(repo_url)
    if parsed.netloc.lower() != "github.com":
        return NormalizedRepoTarget(repo_url=repo_url, branch=branch)

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include both owner and repository name.")

    owner, repo = parts[0], parts[1].removesuffix(".git")
    normalized_url = f"{parsed.scheme or 'https'}://github.com/{owner}/{repo}.git"

    derived_branch = branch
    if len(parts) >= 4 and parts[2] in {"blob", "tree"} and not derived_branch:
        derived_branch = parts[3]

    return NormalizedRepoTarget(repo_url=normalized_url, branch=derived_branch)
