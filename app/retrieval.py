"""Vector store indexing and similarity retrieval backed by ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import chromadb

from app.ingestion import DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved knowledge chunk with similarity distance and source metadata."""

    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    distance: float
    page_number: int | None = None

    def citation_label(self) -> str:
        """Return human-readable citation tag."""
        if self.page_number is not None:
            return f"[{self.source_file} | p.{self.page_number} | chunk {self.chunk_index}]"
        return f"[{self.source_file} | chunk {self.chunk_index}]"


class KnowledgeStore:
    """Persistent or ephemeral ChromaDB vector store for RAG chunks."""

    def __init__(
        self,
        persist_dir: str | None = "./chroma_db",
        collection_name: str = "knowledge_base",
    ) -> None:
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return self._collection.count()

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Upsert DocumentChunk items into the vector store; return count added."""
        if not chunks:
            return 0

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
        return len(chunks)

    def list_sources(self) -> list[str]:
        """Return sorted unique source filenames currently indexed."""
        total = self.count()
        if total == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        sources = {
            str(m["source_file"])
            for m in metadatas
            if isinstance(m, dict) and "source_file" in m
        }
        return sorted(sources)

    def clear(self) -> None:
        """Delete and recreate the collection (useful for reset/tests)."""
        name = self._collection.name
        self._client.delete_collection(name=name)
        self._collection = self._client.get_or_create_collection(name=name)

    def query(
        self,
        question: str,
        top_k: int = 4,
        max_distance: float = 1.5,
    ) -> list[RetrievedChunk]:
        """Query the vector store and return chunks within max_distance."""
        if not question or not question.strip():
            return []

        total = self.count()
        if total == 0:
            return []

        n_results = min(max(1, top_k), total)
        raw = self._collection.query(
            query_texts=[question.strip()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is None or float(dist) > max_distance:
                continue
            meta_dict = meta if isinstance(meta, dict) else {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(meta_dict.get("chunk_id", "")),
                    text=str(doc),
                    source_file=str(meta_dict.get("source_file", "unknown")),
                    chunk_index=int(meta_dict.get("chunk_index", 0)),
                    distance=round(float(dist), 4),
                    page_number=(
                        int(meta_dict["page_number"])
                        if "page_number" in meta_dict and meta_dict["page_number"] is not None
                        else None
                    ),
                )
            )
        return results
