from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.utils.files import is_probably_binary, should_skip_directory, should_skip_file

logger = logging.getLogger(__name__)


LANGUAGE_NAMES: dict[str, str] = {
    ".ipynb": "MARKDOWN",
    ".py": "PYTHON",
    ".js": "JS",
    ".ts": "JS",
    ".jsx": "JS",
    ".tsx": "JS",
    ".java": "JAVA",
    ".go": "GO",
    ".rs": "RUST",
    ".cpp": "CPP",
    ".c": "CPP",
    ".cs": "CSHARP",
    ".php": "PHP",
    ".rb": "RUBY",
    ".scala": "SCALA",
    ".kt": "KOTLIN",
    ".md": "MARKDOWN",
}

GENERIC_SYMBOL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(async\s*)?\(", re.MULTILINE),
    re.compile(r"^\s*export\s+async\s+function\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
]


@dataclass(slots=True)
class ChunkingStats:
    """Basic ingestion counters for reporting and observability."""

    total_files: int = 0
    skipped_files: int = 0


class CodeChunker:
    """Parse repository files into retrieval-friendly LangChain documents."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        max_file_size_bytes: int,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size_bytes = max_file_size_bytes
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_repository(
        self,
        repo_path: Path,
        repo_id: str,
        repo_url: str,
    ) -> tuple[list[Document], ChunkingStats]:
        """Walk the repository tree and chunk supported source files."""
        stats = ChunkingStats()
        documents: list[Document] = []

        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            if should_skip_directory(file_path.parts):
                stats.skipped_files += 1
                continue
            if should_skip_file(file_path):
                stats.skipped_files += 1
                continue
            if file_path.stat().st_size > self.max_file_size_bytes:
                stats.skipped_files += 1
                continue

            raw_bytes = file_path.read_bytes()
            if is_probably_binary(raw_bytes):
                stats.skipped_files += 1
                continue

            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content = raw_bytes.decode("latin-1")
                except UnicodeDecodeError:
                    stats.skipped_files += 1
                    continue

            relative_path = file_path.relative_to(repo_path).as_posix()
            file_documents = self._chunk_file(
                content=content,
                relative_path=relative_path,
                repo_id=repo_id,
                repo_url=repo_url,
            )
            if file_documents:
                stats.total_files += 1
                documents.extend(file_documents)
            else:
                stats.skipped_files += 1

        return documents, stats

    def _chunk_file(
        self,
        content: str,
        relative_path: str,
        repo_id: str,
        repo_url: str,
    ) -> list[Document]:
        """Chunk a single file while preserving useful metadata."""
        suffix = Path(relative_path).suffix.lower()
        if suffix == ".ipynb":
            return self._chunk_notebook_file(content, relative_path, repo_id, repo_url)
        if suffix == ".py":
            return self._chunk_python_file(content, relative_path, repo_id, repo_url)
        return self._chunk_file_fallback(content, relative_path, repo_id, repo_url)

    def _chunk_notebook_file(
        self,
        content: str,
        relative_path: str,
        repo_id: str,
        repo_url: str,
    ) -> list[Document]:
        """Convert a Jupyter notebook into chunkable text blocks by cell."""
        try:
            notebook = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid notebook file: %s", relative_path)
            return []

        cells = notebook.get("cells", [])
        if not isinstance(cells, list) or not cells:
            return []

        documents: list[Document] = []
        virtual_line = 1

        for cell_index, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source", [])
            if isinstance(source, list):
                cell_text = "".join(source)
            else:
                cell_text = str(source)

            cell_text = cell_text.strip()
            if not cell_text:
                continue

            header = f"[notebook_cell type={cell_type} index={cell_index}]"
            prepared_text = f"{header}\n{cell_text}"
            symbol_name = None
            if cell_type == "code":
                symbol_name = self._infer_symbol_name(cell_text)

            if len(prepared_text) <= self.chunk_size + 300:
                end_line = virtual_line + prepared_text.count("\n")
                documents.append(
                    Document(
                        page_content=prepared_text,
                        metadata={
                            "repo_id": repo_id,
                            "repo_url": repo_url,
                            "file_path": relative_path,
                            "chunk_index": cell_index,
                            "symbol_name": symbol_name,
                            "cell_index": cell_index,
                            "cell_type": cell_type,
                            "start_line": virtual_line,
                            "end_line": end_line,
                        },
                    )
                )
                virtual_line = end_line + 2
                continue

            splitter = self._splitter_for_suffix(".ipynb")
            subchunks = splitter.split_text(prepared_text)
            for sub_index, subchunk in enumerate(subchunks):
                sub_start = virtual_line
                sub_end = sub_start + subchunk.count("\n")
                documents.append(
                    Document(
                        page_content=subchunk,
                        metadata={
                            "repo_id": repo_id,
                            "repo_url": repo_url,
                            "file_path": relative_path,
                            "chunk_index": f"{cell_index}.{sub_index}",
                            "symbol_name": symbol_name,
                            "cell_index": cell_index,
                            "cell_type": cell_type,
                            "start_line": sub_start,
                            "end_line": sub_end,
                        },
                    )
                )
                virtual_line = sub_end + 1
            virtual_line += 1

        return documents

    def _chunk_python_file(
        self,
        content: str,
        relative_path: str,
        repo_id: str,
        repo_url: str,
    ) -> list[Document]:
        """Prefer AST-aware chunking for Python to preserve function boundaries."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._chunk_file_fallback(content, relative_path, repo_id, repo_url)

        lines = content.splitlines()
        documents: list[Document] = []
        top_level_nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        if not top_level_nodes:
            return self._chunk_file_fallback(content, relative_path, repo_id, repo_url)

        for index, node in enumerate(top_level_nodes):
            start_line = getattr(node, "lineno", 1)
            end_line = getattr(node, "end_lineno", start_line)
            snippet = "\n".join(lines[start_line - 1 : end_line])
            if len(snippet) <= self.chunk_size + 300:
                documents.append(
                    Document(
                        page_content=snippet,
                        metadata={
                            "repo_id": repo_id,
                            "repo_url": repo_url,
                            "file_path": relative_path,
                            "chunk_index": index,
                            "symbol_name": getattr(node, "name", None),
                            "symbol_type": type(node).__name__,
                            "start_line": start_line,
                            "end_line": end_line,
                        },
                    )
                )
                continue

            splitter = self._splitter_for_suffix(".py")
            subchunks = splitter.split_text(snippet)
            offset = start_line
            for sub_index, subchunk in enumerate(subchunks):
                sub_start = offset
                sub_end = sub_start + subchunk.count("\n")
                offset = sub_end + 1
                documents.append(
                    Document(
                        page_content=subchunk,
                        metadata={
                            "repo_id": repo_id,
                            "repo_url": repo_url,
                            "file_path": relative_path,
                            "chunk_index": f"{index}.{sub_index}",
                            "symbol_name": getattr(node, "name", None),
                            "symbol_type": type(node).__name__,
                            "start_line": sub_start,
                            "end_line": sub_end,
                        },
                    )
                )

        return documents

    def _chunk_file_fallback(
        self,
        content: str,
        relative_path: str,
        repo_id: str,
        repo_url: str,
    ) -> list[Document]:
        """Use recursive chunking when parsing fails or AST chunking is not applicable."""
        suffix = Path(relative_path).suffix.lower()
        splitter = self._splitter_for_suffix(suffix)
        chunks = splitter.split_text(content)
        documents: list[Document] = []
        running_offset = 0

        for index, chunk in enumerate(chunks):
            start_at = content.find(chunk, running_offset)
            if start_at < 0:
                start_at = running_offset
            line_start = content[:start_at].count("\n") + 1
            line_end = line_start + chunk.count("\n")
            running_offset = start_at + len(chunk)
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "repo_id": repo_id,
                        "repo_url": repo_url,
                        "file_path": relative_path,
                        "chunk_index": index,
                        "symbol_name": self._infer_symbol_name(chunk),
                        "start_line": line_start,
                        "end_line": line_end,
                    },
                )
            )
        return documents

    def _splitter_for_suffix(self, suffix: str) -> RecursiveCharacterTextSplitter:
        """Return a language-aware splitter when LangChain supports the extension."""
        language_name = LANGUAGE_NAMES.get(suffix)
        if language_name is None or not hasattr(Language, language_name):
            return self.default_splitter
        language = getattr(Language, language_name)
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @staticmethod
    def _infer_symbol_name(chunk: str) -> str | None:
        """Extract a likely function or class name from a code chunk."""
        for pattern in GENERIC_SYMBOL_PATTERNS:
            match = pattern.search(chunk)
            if match:
                return match.group(1)
        return None
