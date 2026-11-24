"""Tests for ChromaDB KnowledgeStore indexing and retrieval."""

from pathlib import Path

from app.ingestion import ingest_file
from app.retrieval import KnowledgeStore


def test_retrieval_returns_relevant_evidence(tmp_path: Path) -> None:
    store = KnowledgeStore(persist_dir=str(tmp_path / "chroma"), collection_name="test_ret")

    sample_dir = Path("data/sample-knowledge")
    for file_path in sorted(sample_dir.iterdir()):
        chunks = ingest_file(file_path.name, file_path.read_bytes())
        store.add_chunks(chunks)

    assert store.count() >= 3
    assert set(store.list_sources()) == {
        "deployment_policy.md",
        "incident_response.txt",
        "security_onboarding.pdf",
    }

    results = store.query("What is the response SLA for a SEV-1 incident?", top_k=2, max_distance=2.0)
    assert len(results) >= 1
    assert any("incident_response.txt" == r.source_file for r in results)
    assert any("10 minutes" in r.text for r in results)


def test_retrieval_on_empty_store_returns_empty(tmp_path: Path) -> None:
    store = KnowledgeStore(persist_dir=str(tmp_path / "empty_chroma"), collection_name="empty")
    assert store.count() == 0
    assert store.query("Anything?") == []
