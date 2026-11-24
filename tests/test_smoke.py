"""End-to-end smoke test exercising ingestion, indexing, retrieval, and Q&A."""

import json
from pathlib import Path

from app.config import Settings
from app.crew import ABSTENTION_RESPONSE, AgenticRAGPipeline
from app.ingestion import ingest_file
from app.observability import ObservabilityManager
from app.retrieval import KnowledgeStore, RetrievedChunk


def _smoke_fake_llm(question: str, chunks: list[RetrievedChunk]) -> str:
    if "parental leave" in question.lower():
        return ABSTENTION_RESPONSE
    citations = " ".join(c.citation_label() for c in chunks[:2])
    snippets = " ".join(c.text.splitlines()[0] for c in chunks[:2])
    return f"{snippets} {citations}"


def test_end_to_end_smoke_with_evaluation_questions(tmp_path: Path) -> None:
    store = KnowledgeStore(persist_dir=str(tmp_path / "smoke_db"), collection_name="smoke")

    for file_path in sorted(Path("data/sample-knowledge").iterdir()):
        chunks = ingest_file(file_path.name, file_path.read_bytes())
        store.add_chunks(chunks)

    settings = Settings(LANGFUSE_ENABLED=False, MAX_DISTANCE_THRESHOLD=2.0, TOP_K=4)
    pipeline = AgenticRAGPipeline(
        store=store,
        settings=settings,
        observability=ObservabilityManager(settings=settings),
        fake_llm_fn=_smoke_fake_llm,
    )

    questions = [
        json.loads(line)
        for line in Path("evaluation/questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(questions) >= 5

    for item in questions:
        resp = pipeline.ask(item["question"], session_id="smoke-session")
        assert resp.request_id
        assert resp.latency_ms >= 0.0
        if item["category"] == "unanswerable":
            assert resp.abstained is True
            assert resp.answer == ABSTENTION_RESPONSE
        else:
            assert resp.abstained is False
            assert len(resp.citations) >= 1
