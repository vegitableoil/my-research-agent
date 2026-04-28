"""RAG Indexer — chunk, embed, and upsert documents into the vector store."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from ids_agent.rag.indices import INDICES, IndexConfig


class VectorStore(Protocol):
    async def upsert(self, index_name: str, documents: list[dict]) -> None: ...
    async def delete(self, index_name: str, ids: list[str]) -> None: ...


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


class RAGIndexer:
    """Chunks a document and upserts embeddings into the configured vector store."""

    def __init__(self, vector_store: VectorStore, embedding_fn: Any) -> None:
        self.store = vector_store
        self.embed = embedding_fn

    async def index_paper(
        self,
        index_name: str,
        paper_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        config: IndexConfig = INDICES.get(index_name, INDICES["papers_global"])
        chunks = self._chunk(text, paper_id, metadata, config.chunk_size, config.chunk_overlap)
        embeddings = await self.embed([c.text for c in chunks], model=config.embedding_model)

        documents = [
            {
                "id": c.chunk_id,
                "text": c.text,
                "embedding": emb,
                "metadata": c.metadata,
            }
            for c, emb in zip(chunks, embeddings)
        ]
        await self.store.upsert(index_name=index_name, documents=documents)

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk(
        text: str,
        paper_id: str,
        metadata: dict,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk]:
        """Section-aware chunker: splits on section headers first, then by token budget."""
        sections = re.split(r"\n(?=#{1,3} |\d+\.\s+[A-Z])", text)
        chunks: list[Chunk] = []
        idx = 0
        for section in sections:
            words = section.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append(Chunk(
                    chunk_id=f"{paper_id}_chunk_{idx}",
                    text=chunk_text,
                    metadata={**metadata, "paper_id": paper_id, "chunk_index": idx},
                ))
                idx += 1
                start += chunk_size - overlap
        return chunks
