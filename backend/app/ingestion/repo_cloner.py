from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RepoCloner:
    """Clone Git repositories into the local raw data directory."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def clone_or_update(
        self,
        repo_url: str,
        repo_id: str,
        branch: str | None = None,
        force_refresh: bool = False,
    ) -> Path:
        """Clone a repository or refresh it by recloning."""
        target_dir = self.settings.raw_repos_dir / repo_id

        if force_refresh and target_dir.exists():
            shutil.rmtree(target_dir)

        if target_dir.exists():
            logger.info("Repository already present repo_id=%s path=%s", repo_id, target_dir)
            return target_dir

        command = ["git", "clone", "--depth", "1"]
        if branch:
            command.extend(["--branch", branch])
        command.extend([repo_url, str(target_dir)])

        logger.info("Cloning repository repo_id=%s", repo_id)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to clone repository: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return target_dir
