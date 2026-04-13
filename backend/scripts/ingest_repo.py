from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging import configure_logging
from app.models.schemas import IngestRepoRequest
from app.services.repository_service import RepositoryService


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline repository ingestion."""
    parser = argparse.ArgumentParser(description="Ingest a GitHub repository into FAISS.")
    parser.add_argument("repo_url", help="GitHub repository URL")
    parser.add_argument("--branch", default=None, help="Optional branch to clone")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Delete any existing local clone and index before re-ingesting",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ingestion workflow and print a machine-readable summary."""
    configure_logging()
    args = parse_args()

    service = RepositoryService()
    result = service.ingest_repository(
        IngestRepoRequest(
            repo_url=args.repo_url,
            branch=args.branch,
            force_refresh=args.force_refresh,
        )
    )
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
