"""Tests for AgenticRAGPipeline guardrails, citations, and deterministic fake LLM."""

from pathlib import Path

from app.config import Settings
from app.crew import ABSTENTION_RESPONSE, AgenticRAGPipeline
from app.ingestion import ingest_file
from app.observability import ObservabilityManager
from app.retrieval import KnowledgeStore, RetrievedChunk


def _deterministic_fake_llm(question: str, chunks: list[RetrievedChunk]) -> str:
    if "parental leave" in question.lower():
        return ABSTENTION_RESPONSE
    top = chunks[0]
    return f"Based on the knowledge base: {top.text[:120]} {top.citation_label()}"


def test_pipeline_returns_cited_answer_for_known_question(tmp_path: Path) -> None:
    store = KnowledgeStore(persist_dir=str(tmp_path / "chroma"), collection_name="crew_test")
    md_path = Path("data/sample-knowledge/deployment_policy.md")
    store.add_chunks(ingest_file(md_path.name, md_path.read_bytes()))

    settings = Settings(LANGFUSE_ENABLED=False, MAX_DISTANCE_THRESHOLD=2.0)
    obs = ObservabilityManager(settings=settings)
    pipeline = AgenticRAGPipeline(
        store=store,
        settings=settings,
        observability=obs,
        fake_llm_fn=_deterministic_fake_llm,
    )

    resp = pipeline.ask("When are production deployments allowed?")
    assert resp.abstained is False
    assert len(resp.citations) >= 1
    assert "[deployment_policy.md | chunk" in resp.answer


def test_pipeline_abstains_on_empty_store(tmp_path: Path) -> None:
    store = KnowledgeStore(persist_dir=str(tmp_path / "empty"), collection_name="empty_crew")
    settings = Settings(LANGFUSE_ENABLED=False)
    pipeline = AgenticRAGPipeline(
        store=store,
        settings=settings,
        observability=ObservabilityManager(settings=settings),
        fake_llm_fn=_deterministic_fake_llm,
    )

    resp = pipeline.ask("What is the deployment policy?")
    assert resp.abstained is True
    assert resp.answer == ABSTENTION_RESPONSE
    assert resp.citations == []


def test_pipeline_abstains_on_unsupported_question(tmp_path: Path) -> None:
    store = KnowledgeStore(persist_dir=str(tmp_path / "chroma2"), collection_name="unsup")
    txt_path = Path("data/sample-knowledge/incident_response.txt")
    store.add_chunks(ingest_file(txt_path.name, txt_path.read_bytes()))

    settings = Settings(LANGFUSE_ENABLED=False, MAX_DISTANCE_THRESHOLD=2.0)
    pipeline = AgenticRAGPipeline(
        store=store,
        settings=settings,
        observability=ObservabilityManager(settings=settings),
        fake_llm_fn=_deterministic_fake_llm,
    )

    resp = pipeline.ask("What is NovaCloud's parental leave policy?")
    assert resp.abstained is True
    assert resp.answer == ABSTENTION_RESPONSE
    assert resp.citations == []
