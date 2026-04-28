"""RAG Retriever — hybrid dense+sparse search with cross-encoder reranking."""

from __future__ import annotations

from typing import Any, Protocol


class VectorStore(Protocol):
    async def dense_search(self, index_name: str, vector: list[float], k: int) -> list[dict]: ...
    async def sparse_search(self, index_name: str, query: str, k: int) -> list[dict]: ...


class RAGRetriever:
    """
    Performs hybrid retrieval (dense + BM25 sparse) with optional reranking.

    Fusion strategy: Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_fn: Any,
        reranker_fn: Any | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.store = vector_store
        self.embed = embedding_fn
        self.reranker = reranker_fn
        self.rrf_k = rrf_k

    async def retrieve(self, query: str, index: str, k: int = 5) -> list[dict]:
        query_vec = await self.embed([query])
        dense_hits, sparse_hits = await self._parallel_search(query, query_vec[0], index, k * 2)
        fused = self._rrf_fuse(dense_hits, sparse_hits)

        if self.reranker:
            fused = await self.reranker(query=query, candidates=fused[:k * 2])

        return fused[:k]

    # ── private ───────────────────────────────────────────────────────────────

    async def _parallel_search(
        self, query: str, vector: list[float], index: str, k: int
    ) -> tuple[list[dict], list[dict]]:
        # In a real implementation these would run concurrently with asyncio.gather
        dense = await self.store.dense_search(index_name=index, vector=vector, k=k)
        sparse = await self.store.sparse_search(index_name=index, query=query, k=k)
        return dense, sparse

    def _rrf_fuse(self, *result_lists: list[dict]) -> list[dict]:
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}
        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc["id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
                docs[doc_id] = doc
        ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        return [docs[d] for d in ranked]
