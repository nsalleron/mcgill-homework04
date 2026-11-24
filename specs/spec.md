# Product Specification: Agentic RAG Knowledge Assistant

**Version:** 1.0  
**Status:** Draft

## 1. Overview
A Python-based Agentic Retrieval-Augmented Generation (RAG) web application that allows users to upload knowledge documents, index them into a local vector store, and ask natural-language questions answered by a CrewAI workflow with Langfuse observability.

## 2. Functional Requirements

### FR-1: Tab A — Knowledge Upload & Indexing
- **FR-1.1:** Support uploading one or more `.txt`, `.pdf`, and `.md` files.
- **FR-1.2:** Reject unsupported file extensions and empty/whitespace-only files with clear, user-visible error messages (never silently accept).
- **FR-1.3:** Display uploaded file names and per-file processing status (success, chunk count, or error).
- **FR-1.4:** Parse, chunk, embed, and store documents in a persistent local vector store (ChromaDB).
- **FR-1.5:** Preserve source metadata (`source_file`, `chunk_index`) on every stored chunk for citation rendering.

### FR-2: Tab B — Ask Questions
- **FR-2.1:** Provide a natural-language question input.
- **FR-2.2:** Show a visible processing state while the CrewAI workflow executes.
- **FR-2.3:** Return an answer grounded strictly in retrieved chunks, accompanied by source citations / chunk references.
- **FR-2.4:** Return a safe abstention message (*"I could not find this in the uploaded knowledge."*) when evidence is insufficient or no relevant chunks are found.

### FR-3: Agentic Workflow (CrewAI)
- **FR-3.1:** Implement a sequential CrewAI workflow with two agents:
  1. **Retriever/Researcher Agent**: queries the vector store and returns relevant chunks with metadata.
  2. **Answer/Verifier Agent**: drafts the answer using *only* retrieved context, enforces the hallucination guardrail, and attaches citations.

### FR-4: Observability (Langfuse)
- **FR-4.1:** Capture one Langfuse trace per user question with a non-sensitive `session_id` / `request_id`.
- **FR-4.2:** Record nested observations/spans for retrieval, CrewAI crew/agent/task execution, and LLM generation metadata.
- **FR-4.3:** Record latency, status, model name, token usage (when available), and error details on failure.
- **FR-4.4:** Support `LANGFUSE_ENABLED=false` (mock/no-op mode) for unit tests.

## 3. Non-Functional Requirements
- **NFR-1:** Python 3.11+ with `uv` / `pyproject.toml` dependency management.
- **NFR-2:** Only `OPENAI_API_KEY` required for LLM calls (`gpt-4o-mini` default), with local embeddings so indexing works offline.
- **NFR-3:** Secrets stored only in `.env`; `.env.example` documents variable names only.
