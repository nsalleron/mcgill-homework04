# Implementation Tasks

## Phase 1: Project Setup & Skills
- [ ] **Task 1:** Configure `pyproject.toml`, `.env.example`, and install Python dependencies (`crewai`, `langfuse`, `chromadb`, `pypdf`, `streamlit`, `pydantic-settings`, `pytest`).
- [ ] **Task 2:** Install official CrewAI and Langfuse skills into `.agents/skills/`.

## Phase 2: Core Application Modules
- [ ] **Task 3:** Implement `app/config.py` for OpenAI, Langfuse, ChromaDB, and chunking settings.
- [ ] **Task 4:** Implement `app/ingestion.py` (file validation, `.txt`/`.md`/`.pdf` parsing with `page_number`, chunking, `chunk_id` metadata preservation).
- [ ] **Task 5:** Implement `app/retrieval.py` (`KnowledgeStore` with ChromaDB persistence, empty-index guard, and `MAX_DISTANCE_THRESHOLD` filtering).
- [ ] **Task 6:** Implement `app/observability.py` (Langfuse trace/span instrumentation + disabled/mock mode).
- [ ] **Task 7:** Implement `app/crew.py` (CrewAI 2-agent sequential workflow, hallucination guardrail, and fake-LLM test support).
- [ ] **Task 8:** Implement `app/ui.py` (Streamlit 2-tab UI with upload status, processing state, citations, and trace ID).

## Phase 3: Sample Data, Evaluation & Tests
- [ ] **Task 9:** Create `data/sample-knowledge/` documents (`.md`, `.txt`, `.pdf`) and `evaluation/` dataset (`questions.jsonl`, `expected_answers.jsonl`).
- [ ] **Task 10:** Implement automated tests in `tests/` (`test_ingestion.py`, `test_retrieval.py`, `test_observability.py`, `test_crew.py`, `test_smoke.py`) and verify all pass.

## Phase 4: Evaluation Skill & Documentation
- [ ] **Task 11:** Create `.agents/skills/rag-evaluation-report/SKILL.md` (`user-invocable: true`, `disable-model-invocation: true`).
- [ ] **Task 12:** Write `README.md` with setup, Langfuse local/cloud instructions, manual test procedure, and evaluation skill usage.
