# Project Constitution: Agentic RAG Knowledge Assistant

## Core Principles

1. **Groundedness Over Speculation**
   - The system must never fabricate facts or extrapolate beyond retrieved evidence.
   - When evidence is absent or insufficient, the assistant must explicitly abstain (e.g., *"I could not find this in the uploaded knowledge."*).

2. **Specification-First Development**
   - Requirements, architecture, and tasks are defined in `specs/` prior to implementation and updated when design decisions evolve.

3. **Observability-First Architecture**
   - Every user question triggers a structured Langfuse trace capturing nested agent, task, retrieval, and LLM spans, latency, token usage (when available), and errors.
   - Observability must support a clean disabled/mock mode for deterministic offline testing.

4. **Secret & Privacy Hygiene**
   - Credentials (`OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) live exclusively in `.env` (gitignored) and are never committed.
   - Traces must avoid logging secrets or unnecessary personal data.

5. **Deterministic Testability & Modularity**
   - Ingestion, retrieval, agent orchestration, observability, and UI are decoupled modules.
   - Core workflows must be testable without live external APIs using test doubles.
