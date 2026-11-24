# Technical Implementation Plan

**Version:** 1.0

## 1. Technology Stack
- **Language:** Python 3.11+
- **UI:** Streamlit (`st.tabs` for Tab A & Tab B, `st.status` for processing indicator)
- **Agent Orchestration:** CrewAI (sequential process, 2 agents, custom retrieval tool)
- **LLM Provider:** OpenAI (`gpt-4o-mini` via `OPENAI_API_KEY`)
- **Vector Store & Embeddings:** ChromaDB (local `PersistentClient` + built-in `all-MiniLM-L6-v2` embeddings)
- **PDF Parsing:** `pypdf`
- **Observability:** Langfuse Python SDK (`langfuse`)
- **Testing:** `pytest`

## 2. Module Architecture
- `app/config.py`: Pydantic-based environment configuration.
- `app/ingestion.py`: File validation (`.txt`, `.pdf`, `.md`, non-empty), text extraction, chunking with overlap, and metadata attachment.
- `app/retrieval.py`: `KnowledgeStore` wrapping ChromaDB for indexing and similarity search.
- `app/observability.py`: `ObservabilityManager` wrapping Langfuse traces/spans with a safe no-op fallback when disabled.
- `app/crew.py`: CrewAI `Retriever/Researcher` and `Answer/Verifier` agents, tasks, hallucination guardrail, and deterministic fake-LLM test hook.
- `app/ui.py`: Two-tab Streamlit application.

## 3. Trace Hierarchy (Langfuse)
```
Trace: rag-qa-query (input=question, session_id, metadata={request_id})
├── Span: retrieval (input=question, output=chunk_metadata, latency_ms)
└── Span: crewai-workflow (model, status, latency_ms)
    ├── Span: retriever-agent-task
    └── Span: answer-verifier-agent-task (generation metadata, tokens)
```
