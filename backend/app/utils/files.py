from __future__ import annotations

from pathlib import Path
from typing import Iterable


SKIP_DIRECTORIES = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".turbo",
    "vendor",
    "target",
    ".idea",
    ".vscode",
}

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".jar",
    ".lock",
    ".min.js",
    ".min.css",
}

SUPPORTED_SUFFIXES = {
    ".ipynb",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".cs",
    ".php",
    ".rb",
    ".scala",
    ".kt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
}


def should_skip_directory(parts: Iterable[str]) -> bool:
    """Return true when any path segment belongs to a known vendor/build directory."""
    return any(part in SKIP_DIRECTORIES for part in parts)


def should_skip_file(file_path: Path) -> bool:
    """Return true when a file should be ignored during ingestion."""
    suffix = file_path.suffix.lower()
    full_name = file_path.name.lower()
    if any(full_name.endswith(ignored) for ignored in SKIP_SUFFIXES):
        return True
    return suffix not in SUPPORTED_SUFFIXES


def is_probably_binary(raw_bytes: bytes) -> bool:
    """Heuristic binary detector to avoid embedding non-text files."""
    if not raw_bytes:
        return False
    return b"\x00" in raw_bytes[:2048]
