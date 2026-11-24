"""Streamlit two-tab UI for the Agentic RAG Knowledge Assistant."""

from __future__ import annotations

from pathlib import Path
import sys
import uuid

import streamlit as st

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.config import get_settings
from app.crew import AgenticRAGPipeline
from app.ingestion import IngestionValidationError, ingest_file
from app.observability import ObservabilityManager
from app.retrieval import KnowledgeStore


def _get_store() -> KnowledgeStore:
    settings = get_settings()
    if "knowledge_store" not in st.session_state:
        st.session_state["knowledge_store"] = KnowledgeStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
    return st.session_state["knowledge_store"]


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def main() -> None:
    st.set_page_config(
        page_title="Agentic RAG Knowledge Assistant",
        page_icon="📚",
        layout="wide",
    )
    settings = get_settings()
    store = _get_store()
    session_id = _get_session_id()

    st.title("Agentic RAG Knowledge Assistant")
    st.caption("Powered by CrewAI, ChromaDB, OpenAI, and Langfuse Observability")

    with st.sidebar:
        st.header("System Status")
        st.write(f"**Provider:** `{settings.llm_provider}`")
        st.write(f"**Model:** `{settings.active_model_name}`")
        if settings.llm_provider.lower() == "bedrock":
            st.write(f"**AWS Region:** `{settings.aws_region}`")
            if settings.aws_profile:
                st.write(f"**AWS Profile:** `{settings.aws_profile}`")
        else:
            st.write(
                f"**OpenAI Key:** {'Configured' if settings.openai_api_key else 'Missing'}"
            )
        st.write(
            f"**Langfuse:** {'Enabled' if settings.langfuse_configured else 'Disabled / Mock'}"
        )
        st.metric("Indexed Chunks", store.count())

        sources = store.list_sources()
        if sources:
            st.write("**Indexed Files:**")
            for src in sources:
                st.write(f"- `{src}`")

        if st.button("Clear Knowledge Base", type="secondary"):
            store.clear()
            st.success("Knowledge base cleared.")
            st.rerun()

    tab_upload, tab_ask = st.tabs(
        ["Tab A: Knowledge Upload & Indexing", "Tab B: Ask Questions"]
    )

    with tab_upload:
        st.subheader("Upload and Index Knowledge Files")
        st.write("Supported formats: `.txt`, `.pdf`, `.md` (non-empty files).")

        uploaded_files = st.file_uploader(
            "Choose one or more files",
            accept_multiple_files=True,
            key="rag_file_uploader",
        )

        if st.button("Process & Index Files", type="primary"):
            if not uploaded_files:
                st.warning("Please select at least one file before indexing.")
            else:
                results_table = []
                total_chunks = 0
                errors: list[str] = []

                with st.status("Validating, parsing, and indexing files...", expanded=True):
                    for uf in uploaded_files:
                        raw_bytes = uf.getvalue()
                        try:
                            chunks = ingest_file(
                                filename=uf.name,
                                raw_bytes=raw_bytes,
                                chunk_size=settings.chunk_size,
                                chunk_overlap=settings.chunk_overlap,
                            )
                            added = store.add_chunks(chunks)
                            total_chunks += added
                            results_table.append(
                                {
                                    "File": uf.name,
                                    "Status": "Indexed",
                                    "Chunks": added,
                                    "Detail": "OK",
                                }
                            )
                        except IngestionValidationError as exc:
                            err_msg = str(exc)
                            errors.append(err_msg)
                            results_table.append(
                                {
                                    "File": uf.name,
                                    "Status": "Error",
                                    "Chunks": 0,
                                    "Detail": err_msg,
                                }
                            )

                st.dataframe(results_table, use_container_width=True)
                if total_chunks > 0:
                    st.success(
                        f"Successfully indexed {total_chunks} chunk(s). Total chunks in store: {store.count()}."
                    )
                if errors:
                    st.error("Some files failed validation:\n- " + "\n- ".join(errors))

    with tab_ask:
        st.subheader("Ask a Question")
        question = st.text_input(
            "Enter your natural-language question:",
            placeholder="e.g., What is the deployment rollback SLA?",
        )

        if st.button("Ask Assistant", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                obs = ObservabilityManager(settings=settings)
                pipeline = AgenticRAGPipeline(
                    store=store,
                    settings=settings,
                    observability=obs,
                )
                try:
                    with st.status(
                        "Running CrewAI Retriever & Answer Verifier agents...",
                        expanded=True,
                    ) as status:
                        st.write("1. Querying vector store for relevant evidence...")
                        response = pipeline.ask(question, session_id=session_id)
                        st.write("2. Verifying groundedness and formatting citations...")
                        status.update(label="Completed", state="complete", expanded=False)

                    if response.abstained:
                        st.warning(response.answer)
                    else:
                        st.markdown("### Answer")
                        st.markdown(response.answer)

                    if response.citations:
                        with st.expander(
                            f"Source Citations ({len(response.citations)} retrieved chunks)"
                        ):
                            for chunk in response.citations:
                                st.markdown(
                                    f"**{chunk.citation_label()}** (distance: `{chunk.distance}`)"
                                )
                                st.code(chunk.text, language=None)

                    meta_cols = st.columns(3)
                    meta_cols[0].caption(f"Request ID: `{response.request_id}`")
                    meta_cols[1].caption(f"Latency: `{response.latency_ms} ms`")
                    meta_cols[2].caption(
                        f"Langfuse Trace: `{response.trace_id or 'disabled'}`"
                    )
                except Exception as exc:
                    st.error(f"Workflow failed: {exc}")


if __name__ == "__main__":
    main()
