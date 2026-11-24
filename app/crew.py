"""CrewAI sequential RAG workflow with hallucination guardrails and Langfuse tracing."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.observability import ObservabilityManager
from app.retrieval import KnowledgeStore, RetrievedChunk

ABSTENTION_RESPONSE = "I could not find this in the uploaded knowledge."


@dataclass
class RAGResponse:
    """Structured result returned by the Agentic RAG workflow."""

    answer: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    trace_id: str | None = None
    request_id: str = ""
    latency_ms: float = 0.0
    model: str = ""


class _RetrievalInput(BaseModel):
    query: str = Field(..., description="The natural-language question to search for.")


class KnowledgeRetrievalTool(BaseTool):
    """CrewAI tool that queries the ChromaDB KnowledgeStore and preserves metadata."""

    name: str = "knowledge_base_search"
    description: str = (
        "Search the uploaded knowledge base for relevant text chunks and their source metadata."
    )
    args_schema: type[BaseModel] = _RetrievalInput

    store: Any = Field(exclude=True)
    top_k: int = 4
    max_distance: float = 1.5
    captured_chunks: list[RetrievedChunk] = Field(default_factory=list, exclude=True)

    def _run(self, query: str) -> str:
        chunks = self.store.query(
            question=query,
            top_k=self.top_k,
            max_distance=self.max_distance,
        )
        self.captured_chunks.clear()
        self.captured_chunks.extend(chunks)

        if not chunks:
            return "NO_EVIDENCE_FOUND"

        formatted = []
        for chunk in chunks:
            formatted.append(
                f"{chunk.citation_label()} (distance={chunk.distance}):\n{chunk.text}"
            )
        return "\n\n---\n\n".join(formatted)


def _extract_token_usage(crew_output: Any) -> dict[str, int] | None:
    """Safely extract token usage dict from a CrewOutput object if present."""
    usage = getattr(crew_output, "token_usage", None)
    if usage is None:
        return None
    result: dict[str, int] = {}
    for src_key, dst_key in (
        ("prompt_tokens", "input"),
        ("completion_tokens", "output"),
        ("total_tokens", "total"),
    ):
        val = getattr(usage, src_key, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(src_key)
        if isinstance(val, int) and val >= 0:
            result[dst_key] = val
    return result or None


class AgenticRAGPipeline:
    """Orchestrates retrieval, CrewAI agents, hallucination guardrails, and Langfuse."""

    def __init__(
        self,
        store: KnowledgeStore,
        settings: Settings | None = None,
        observability: ObservabilityManager | None = None,
        fake_llm_fn: Callable[[str, list[RetrievedChunk]], str] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.obs = observability or ObservabilityManager(settings=self.settings)
        self.fake_llm_fn = fake_llm_fn

    def ask(self, question: str, session_id: str | None = None) -> RAGResponse:
        """Run the end-to-end traced Agentic RAG Q&A workflow for a user question."""
        cleaned_q = (question or "").strip()
        if not cleaned_q:
            raise ValueError("Question cannot be empty.")

        request_id = self.obs.new_request_id()
        start = time.perf_counter()
        model_name = (
            "fake-deterministic-llm"
            if self.fake_llm_fn
            else self.settings.active_model_name
        )

        with self.obs.observation(
            "rag-qa-query",
            as_type="span",
            input={"question": cleaned_q},
            metadata={
                "request_id": request_id,
                "session_id": session_id or "default-session",
                "model": model_name,
            },
        ) as root_span:
            trace_id = root_span.trace_id

            # Step 1: Retrieval observation + empty/threshold guardrail
            with self.obs.observation(
                "vector-retrieval",
                as_type="retriever",
                input={"query": cleaned_q, "top_k": self.settings.top_k},
            ) as ret_span:
                initial_chunks = self.store.query(
                    question=cleaned_q,
                    top_k=self.settings.top_k,
                    max_distance=self.settings.max_distance_threshold,
                )
                ret_span.update(
                    output={
                        "chunk_count": len(initial_chunks),
                        "sources": [
                            {
                                "chunk_id": c.chunk_id,
                                "source_file": c.source_file,
                                "chunk_index": c.chunk_index,
                                "distance": c.distance,
                            }
                            for c in initial_chunks
                        ],
                    }
                )

            if not initial_chunks:
                with self.obs.observation(
                    "hallucination-guardrail",
                    as_type="guardrail",
                    input={"reason": "no_relevant_chunks"},
                ) as guard_span:
                    guard_span.update(
                        output={"action": "abstain", "response": ABSTENTION_RESPONSE}
                    )
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                root_span.update(
                    output={"answer": ABSTENTION_RESPONSE, "abstained": True}
                )
                self.obs.flush()
                return RAGResponse(
                    answer=ABSTENTION_RESPONSE,
                    citations=[],
                    abstained=True,
                    trace_id=trace_id,
                    request_id=request_id,
                    latency_ms=latency_ms,
                    model=model_name,
                )

            # Step 2: Deterministic test double path (for unit/smoke tests)
            if self.fake_llm_fn is not None:
                with self.obs.observation(
                    "crewai-workflow-test-double",
                    as_type="agent",
                    input={"question": cleaned_q, "chunks": len(initial_chunks)},
                    model=model_name,
                ) as crew_span:
                    answer_text = self.fake_llm_fn(cleaned_q, initial_chunks).strip()
                    if not answer_text:
                        answer_text = ABSTENTION_RESPONSE
                    abstained = ABSTENTION_RESPONSE.lower() in answer_text.lower()
                    crew_span.update(
                        output={"answer": answer_text, "abstained": abstained},
                        model=model_name,
                    )

                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                root_span.update(output={"answer": answer_text, "abstained": abstained})
                self.obs.flush()
                return RAGResponse(
                    answer=answer_text,
                    citations=[] if abstained else initial_chunks,
                    abstained=abstained,
                    trace_id=trace_id,
                    request_id=request_id,
                    latency_ms=latency_ms,
                    model=model_name,
                )

            # Step 3: Real CrewAI sequential workflow
            provider = self.settings.llm_provider.lower()
            if provider == "bedrock":
                if self.settings.aws_region:
                    os.environ["AWS_DEFAULT_REGION"] = self.settings.aws_region
                    os.environ["AWS_REGION_NAME"] = self.settings.aws_region
                if self.settings.aws_profile.strip():
                    os.environ["AWS_PROFILE"] = self.settings.aws_profile.strip()
                else:
                    os.environ.pop("AWS_PROFILE", None)
                bedrock_model = self.settings.bedrock_model_name
                if not bedrock_model.startswith("bedrock/"):
                    bedrock_model = f"bedrock/{bedrock_model}"
                llm = LLM(
                    model=bedrock_model,
                    region_name=self.settings.aws_region,
                    temperature=0.1,
                )
            else:
                if not self.settings.openai_api_key.strip():
                    raise RuntimeError(
                        "OPENAI_API_KEY is not configured. Set it in your .env file."
                    )
                os.environ["OPENAI_API_KEY"] = self.settings.openai_api_key
                llm = LLM(
                    model=f"openai/{self.settings.openai_model_name}",
                    temperature=0.1,
                )
            retrieval_tool = KnowledgeRetrievalTool(
                store=self.store,
                top_k=self.settings.top_k,
                max_distance=self.settings.max_distance_threshold,
            )

            with self.obs.observation(
                "crewai-workflow",
                as_type="agent",
                input={"question": cleaned_q, "process": "sequential"},
                model=model_name,
            ) as crew_span:
                retriever_agent = Agent(
                    role="Retriever and Researcher",
                    goal=(
                        "Retrieve the most relevant evidence chunks from the knowledge base "
                        "for the user's question without altering source metadata."
                    ),
                    backstory=(
                        "You are a meticulous research specialist who always queries the "
                        "knowledge base tool and returns exact evidence with citations."
                    ),
                    tools=[retrieval_tool],
                    llm=llm,
                    verbose=False,
                    allow_delegation=False,
                )

                verifier_agent = Agent(
                    role="Grounded Answer Verifier",
                    goal=(
                        "Draft a concise, accurate answer using ONLY the retrieved evidence, "
                        "include inline citations, and refuse to speculate when evidence is missing."
                    ),
                    backstory=(
                        "You are a strict fact-checker and hallucination guardrail. You never "
                        "invent facts outside the provided context. If the context does not "
                        f"answer the question, you respond with exactly: '{ABSTENTION_RESPONSE}'"
                    ),
                    llm=llm,
                    verbose=False,
                    allow_delegation=False,
                )

                research_task = Task(
                    description=(
                        f"Use the knowledge_base_search tool to find evidence for:\n"
                        f"Question: {cleaned_q}\n\n"
                        "Return all relevant chunks verbatim along with their citation tags "
                        "(e.g. [filename | chunk X]). If NO_EVIDENCE_FOUND is returned, state that clearly."
                    ),
                    expected_output=(
                        "Verbatim retrieved chunks with their source citation tags, or NO_EVIDENCE_FOUND."
                    ),
                    agent=retriever_agent,
                )

                answer_task = Task(
                    description=(
                        f"Answer the user question using ONLY the retrieved evidence from the previous task.\n"
                        f"Question: {cleaned_q}\n\n"
                        "Rules:\n"
                        "1. Every factual claim must be supported by the retrieved chunks and include inline citations like [filename | chunk X].\n"
                        "2. Distinguish clearly between what is supported by evidence and any uncertainty.\n"
                        f"3. If the retrieved evidence is insufficient or NO_EVIDENCE_FOUND, respond with EXACTLY:\n{ABSTENTION_RESPONSE}"
                    ),
                    expected_output=(
                        "A grounded answer with inline citations, or the exact abstention message if evidence is insufficient."
                    ),
                    agent=verifier_agent,
                    context=[research_task],
                )

                crew = Crew(
                    agents=[retriever_agent, verifier_agent],
                    tasks=[research_task, answer_task],
                    process=Process.sequential,
                    verbose=False,
                )

                with self.obs.observation(
                    "crewai-llm-generation",
                    as_type="generation",
                    input={"question": cleaned_q, "context_chunks": len(initial_chunks)},
                    model=model_name,
                ) as gen_span:
                    crew_result = crew.kickoff()
                    answer_text = str(getattr(crew_result, "raw", crew_result)).strip()
                    if not answer_text:
                        answer_text = ABSTENTION_RESPONSE
                    usage = _extract_token_usage(crew_result)
                    gen_span.update(
                        output={"answer": answer_text},
                        model=model_name,
                        usage_details=usage,
                    )

                used_chunks = retrieval_tool.captured_chunks or initial_chunks
                abstained = ABSTENTION_RESPONSE.lower() in answer_text.lower()
                crew_span.update(
                    output={"answer": answer_text, "abstained": abstained},
                    model=model_name,
                )

            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            root_span.update(output={"answer": answer_text, "abstained": abstained})
            self.obs.flush()

            return RAGResponse(
                answer=answer_text,
                citations=[] if abstained else used_chunks,
                abstained=abstained,
                trace_id=trace_id,
                request_id=request_id,
                latency_ms=latency_ms,
                model=model_name,
            )
