# Agentic RAG Knowledge Assistant (CrewAI + Langfuse)

A specification-first Python Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **ChromaDB**, **CrewAI**, **OpenAI**, and **Langfuse** observability.

## Project Structure

```text
├── app/
│   ├── config.py          # Pydantic settings (.env)
│   ├── ingestion.py       # File validation (.txt/.pdf/.md), parsing, chunking, metadata
│   ├── retrieval.py       # ChromaDB vector store & distance-threshold filtering
│   ├── observability.py   # Langfuse v5 trace/span instrumentation + no-op test mode
│   ├── crew.py            # Sequential CrewAI 2-agent workflow + hallucination guardrail
│   └── ui.py              # Two-tab Streamlit UI
├── data/sample-knowledge/ # Sample .md, .txt, and .pdf files
├── evaluation/            # 5-question evaluation dataset & expected answers
├── specs/                 # Spec Kit artifacts (constitution, spec, plan, tasks)
├── tests/                 # Automated unit & smoke tests (pytest)
├── .agents/skills/        # Official CrewAI & Langfuse skills + rag-evaluation-report
├── .env.example           # Environment variable template (no secrets)
└── pyproject.toml         # Python dependencies & pytest config
```

## 1. Setup & Installation

### Prerequisites
- Python 3.11–3.13 (tested on Python 3.12)
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key (`OPENAI_API_KEY`)
- A Langfuse account ([Langfuse Cloud](https://cloud.langfuse.com) or local Docker instance)

### Install Python Dependencies
```bash
uv sync --extra dev --python 3.12
```

### Configure Environment Variables (Without Exposing Secrets)
1. Copy `.env.example` to `.env` (`.env` is gitignored and must **never** be committed):
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and fill in your credentials:
   ```dotenv
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL_NAME=gpt-4o-mini

   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

## 2. Langfuse Local or Cloud Setup & Trace Verification

### Option A: Langfuse Cloud
1. Create a free project at [cloud.langfuse.com](https://cloud.langfuse.com).
2. Go to **Settings -> API Keys** and generate a key pair (`pk-lf-...`, `sk-lf-...`).
3. Set `LANGFUSE_ENABLED=true` and `LANGFUSE_HOST=https://cloud.langfuse.com` in `.env`.

### Option B: Local Langfuse (Docker)
1. Clone and start the official Langfuse Docker Compose stack:
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse && docker compose up -d
   ```
2. Open `http://localhost:3000`, create a project and API keys, and set `LANGFUSE_HOST=http://localhost:3000` in `.env`.

### Confirming Traces Are Arriving
1. Start the Streamlit UI and ask a question in **Tab B**.
2. Note the **Langfuse Trace ID** displayed under the answer.
3. Open your Langfuse project **Tracing -> Traces** view and locate `rag-qa-query`.
4. Expand the trace to inspect nested observations:
   - `vector-retrieval` (`retriever` span with chunk metadata & distances)
   - `crewai-workflow` (`agent` span)
   - `crewai-llm-generation` (`generation` span with model name & token usage)
   - `hallucination-guardrail` (`guardrail` span when abstaining)

## 3. Running the Application

```bash
uv run streamlit run app/ui.py
```

## 4. Automated Tests

All unit and smoke tests run deterministically offline with `LANGFUSE_ENABLED=false` and a fake LLM test double:

```bash
uv run pytest -v
```

## 5. Repeatable Manual Test Procedure

1. Launch the UI: `uv run streamlit run app/ui.py`.
2. **Tab A (Knowledge Upload & Indexing)**:
   - Upload all three files in `data/sample-knowledge/` (`deployment_policy.md`, `incident_response.txt`, `security_onboarding.pdf`) and click **Process & Index Files**.
   - Verify the status table shows `Indexed` and chunk counts > 0.
   - Upload an empty `.txt` file and an unsupported `.csv` file; verify clear validation errors are shown and neither is indexed.
3. **Tab B (Ask Questions)**:
   - Ask: *"What is the response SLA for a SEV-1 incident?"* -> Verify visible processing status, grounded answer (`10 minutes`), and citation `[incident_response.txt | chunk 0]`.
   - Ask: *"If a Friday hotfix deployment causes a SEV-1 outage, who must approve the deployment and within how many hours must the post-mortem be published?"* -> Verify multi-document synthesis citing both `deployment_policy.md` and `incident_response.txt`.
   - Ask: *"What is NovaCloud's parental leave policy?"* -> Verify safe abstention: `"I could not find this in the uploaded knowledge."`.
4. **Langfuse Verification**: Confirm the trace ID from Tab B appears in Langfuse with nested spans, latency, and status.

## 6. Skills Configuration & Evaluation Report

- **Official Skills Installed:** `.agents/skills/getting-started` (`crewaiinc/skills@getting-started`) and `.agents/skills/langfuse` (`langfuse/skills@langfuse`).
- **Evaluation Skill:** `.agents/skills/rag-evaluation-report/SKILL.md` (`user-invocable: true`, `disable-model-invocation: true`).
- After testing is complete, explicitly invoke `/rag-evaluation-report` to generate `reports/rag-evaluation-report.md`.
