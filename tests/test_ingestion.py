"""Tests for file validation, parsing, chunking, and metadata preservation."""

from pathlib import Path
import pytest

from app.ingestion import (
    IngestionValidationError,
    chunk_text,
    ingest_file,
    ingest_files,
    validate_file,
)


def test_validate_rejects_unsupported_extension() -> None:
    with pytest.raises(IngestionValidationError, match="Unsupported file type"):
        validate_file("malware.exe", b"binary")


def test_validate_rejects_empty_file() -> None:
    with pytest.raises(IngestionValidationError, match="empty"):
        validate_file("notes.txt", b"")


def test_ingest_rejects_whitespace_only_txt() -> None:
    with pytest.raises(IngestionValidationError, match="whitespace"):
        ingest_file("blank.txt", b"   \n\t  ")


def test_ingest_txt_and_md_preserves_metadata() -> None:
    content = ("Paragraph one about deployment windows. " * 25).encode("utf-8")
    chunks = ingest_file("policy.md", content, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.source_file == "policy.md"
        assert chunk.chunk_index == idx
        assert chunk.chunk_id == f"policy.md::chunk_{idx}"
        assert chunk.page_number is None
        meta = chunk.to_metadata()
        assert meta["source_file"] == "policy.md"
        assert meta["chunk_index"] == idx


def test_ingest_sample_pdf_preserves_page_number() -> None:
    pdf_path = Path("data/sample-knowledge/security_onboarding.pdf")
    chunks = ingest_file(pdf_path.name, pdf_path.read_bytes())

    assert len(chunks) >= 1
    assert chunks[0].source_file == "security_onboarding.pdf"
    assert chunks[0].page_number == 1
    assert "HashiCorp Vault" in chunks[0].text


def test_ingest_files_collects_errors_without_crashing() -> None:
    chunks, errors = ingest_files(
        [
            ("good.txt", b"Valid knowledge text."),
            ("bad.csv", b"a,b,c"),
            ("empty.md", b""),
        ]
    )
    assert len(chunks) == 1
    assert len(errors) == 2


def test_chunk_text_invalid_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, chunk_overlap=100)
