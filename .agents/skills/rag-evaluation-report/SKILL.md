---
name: rag-evaluation-report
description: Generate an honest MLOps evaluation report for the Agentic RAG Knowledge Assistant using specs, test results, evaluation dataset, configuration, and Langfuse traces.
user-invocable: true
disable-model-invocation: true
---

# RAG Evaluation Report Skill

> **Invocation Policy:** This skill MUST ONLY be invoked when the user explicitly requests an evaluation report (e.g., `/rag-evaluation-report`). It must NEVER run automatically during agentic coding, document upload, or question answering.

## Purpose

Produce an evidence-based, honest MLOps evaluation report at `reports/rag-evaluation-report.md` grounded in the project's actual artifacts and the evaluation taxonomy from Homework 03 (software quality, ML classification/retrieval metrics, reliability/latency, and operational safety).

## Mandatory Inputs to Read

Before writing the report, read and inspect:
1. **Specifications:** `specs/constitution.md`, `specs/spec.md`, `specs/plan.md`, `specs/tasks.md`
2. **Configuration & Code:** `app/config.py`, `app/ingestion.py`, `app/retrieval.py`, `app/crew.py`, `app/observability.py`
3. **Evaluation Dataset:** `evaluation/questions.jsonl` and `evaluation/expected_answers.jsonl`
4. **Automated Test Results:** Run `uv run pytest -v` (or read the latest verified test output)
5. **Langfuse Trace Summaries:** Inspect any exported Langfuse trace JSON/IDs or live Langfuse CLI/API output *if configured and available*.

## Strict Anti-Fabrication Rule

- **NEVER invent, estimate, or fabricate metric values or Langfuse trace data.**
- If a metric was not directly measured in test outputs, evaluation runs, or Langfuse traces, mark its value as **`Unavailable`** and state **how to collect it** (e.g., running an evaluation harness with ground-truth chunk IDs, enabling Langfuse Cloud/local, or adding an LLM-as-a-Judge evaluator).

## Required Report Structure (`reports/rag-evaluation-report.md`)

1. **Executive Summary & System Configuration**
   - Model (`OPENAI_MODEL_NAME`), chunking (`CHUNK_SIZE`, `CHUNK_OVERLAP`), retrieval (`TOP_K`, `MAX_DISTANCE_THRESHOLD`), and Langfuse status.
2. **Test & Dataset Coverage Summary**
   - `pytest` pass/fail counts and coverage of validation, chunking, retrieval, abstention, citations, observability, and smoke test.
   - Breakdown of `evaluation/questions.jsonl` (single-doc, unanswerable, multi-doc).
3. **Multi-Dimensional Evaluation Metrics (Homework 03 + Homework 04 Taxonomy)**
   Report observed values where measured and `Unavailable` + collection method where not measured:
   - **Retrieval:** Recall@k, Precision@k, Hit Rate@k, MRR, nDCG, context relevance, chunk coverage.
   - **Generation / RAG Quality:** faithfulness/groundedness, answer relevance, context utilization, citation correctness, citation completeness, unsupported-claim rate.
   - **Task Outcome:** correctness against expected answer points, abstention quality, human/rubric score.
   - **Agent Workflow:** task success rate, tool/retrieval failure rate, retries, trace completeness, step latency, end-to-end latency, token usage/cost.
   - **Operations:** upload success rate, indexing latency, query throughput, error rate, resource usage.
   - **Safety & Privacy:** prompt-injection detection rate, data leakage checks, secret exposure (`.env` hygiene), PII handling.
4. **Limitations & Data-Quality Concerns**
   - Small sample corpus size, character-based chunking vs. semantic/heading-aware splitting, absence of reranking, deterministic fake-LLM vs. live OpenAI variance.
5. **Privacy & Security Considerations**
   - Secret isolation (`.env` gitignored), trace payload sanitization, local ChromaDB storage.
6. **Recommended Next Experiments**
   - Concrete, prioritized experiments (e.g., automated RAGAS/Langfuse LLM-as-a-Judge scoring on `evaluation/questions.jsonl`, chunk size ablation, hybrid BM25 + dense retrieval).
