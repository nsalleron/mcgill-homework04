"""Document validation, parsing, chunking, and metadata preservation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class IngestionValidationError(ValueError):
    """Raised when an uploaded file fails validation (type, empty, unreadable)."""


@dataclass(frozen=True)
class DocumentChunk:
    """A single text chunk with source metadata for vector indexing and citations."""

    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    char_count: int
    page_number: int | None = None

    def to_metadata(self) -> dict[str, str | int]:
        """Convert chunk metadata into Chroma-compatible flat dictionary."""
        meta: dict[str, str | int] = {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
        }
        if self.page_number is not None:
            meta["page_number"] = self.page_number
        return meta


def validate_file(filename: str, raw_bytes: bytes) -> str:
    """Validate file extension and non-empty content; return normalized extension."""
    if not filename or not filename.strip():
        raise IngestionValidationError("Filename cannot be empty.")

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionValidationError(
            f"Unsupported file type '{ext or 'none'}' for '{filename}'. Allowed types: {allowed}."
        )

    if not raw_bytes or len(raw_bytes) == 0:
        raise IngestionValidationError(f"File '{filename}' is empty (0 bytes).")

    return ext


def _extract_pages(filename: str, ext: str, raw_bytes: bytes) -> list[tuple[int | None, str]]:
    """Extract (page_number, text) pairs from .txt, .md, or .pdf bytes."""
    if ext in {".txt", ".md"}:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")
        if not text.strip():
            raise IngestionValidationError(f"File '{filename}' contains only whitespace.")
        return [(None, text)]

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise IngestionValidationError(
                f"Failed to parse PDF '{filename}': {exc}"
            ) from exc

        pages: list[tuple[int | None, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append((idx, page_text))

        if not pages:
            raise IngestionValidationError(
                f"PDF '{filename}' contains no extractable text."
            )
        return pages

    raise IngestionValidationError(f"Unsupported file type '{ext}'.")


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks with clean boundary preference."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step

    return chunks


def ingest_file(
    filename: str,
    raw_bytes: bytes,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Validate, parse, and chunk a single file into DocumentChunk records."""
    ext = validate_file(filename, raw_bytes)
    safe_name = Path(filename).name
    pages = _extract_pages(safe_name, ext, raw_bytes)

    all_chunks: list[DocumentChunk] = []
    global_idx = 0
    for page_num, page_text in pages:
        for piece in chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunk_id = f"{safe_name}::chunk_{global_idx}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source_file=safe_name,
                    chunk_index=global_idx,
                    char_count=len(piece),
                    page_number=page_num,
                )
            )
            global_idx += 1

    if not all_chunks:
        raise IngestionValidationError(f"No usable text chunks extracted from '{safe_name}'.")

    return all_chunks


def ingest_files(
    files: Iterable[tuple[str, bytes]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> tuple[list[DocumentChunk], list[str]]:
    """Ingest multiple (filename, bytes) pairs, returning (chunks, error_messages)."""
    collected: list[DocumentChunk] = []
    errors: list[str] = []
    for filename, raw_bytes in files:
        try:
            collected.extend(
                ingest_file(
                    filename,
                    raw_bytes,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        except IngestionValidationError as exc:
            errors.append(str(exc))
    return collected, errors
