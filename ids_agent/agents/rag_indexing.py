"""RAG Indexing Agent — chunk, embed, and upsert paper content."""

from __future__ import annotations

from typing import Any

from ids_agent.agents.base import AgentContext, BaseAgent
from ids_agent.rag.indexer import RAGIndexer


class RAGIndexingAgent(BaseAgent):
    """
    Chunks a paper's full text and upserts it into the appropriate vector index.

    ctx.data inputs:
        summary (dict)         — structured summary (provides metadata)
        full_text (str)        — raw paper text

    ctx.data outputs:
        indexed_paper_id (str)
    """

    rag_indices = []

    def __init__(self, *args: Any, indexer: RAGIndexer, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.indexer = indexer

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        summary: dict = ctx.data["summary"]
        full_text: str = ctx.data.get("full_text", "")
        paper_id: str = summary["paper_id"]

        # Global index for cross-paper retrieval
        await self.indexer.index_paper(
            index_name="papers_global",
            paper_id=paper_id,
            text=full_text,
            metadata=summary,
        )

        # Per-paper index used by BaselineVerifierAgent
        await self.indexer.index_paper(
            index_name=f"paper_{paper_id}",
            paper_id=paper_id,
            text=full_text,
            metadata=summary,
        )

        self.logger.info("Indexed paper %s into vector DB", paper_id)
        return {"indexed_paper_id": paper_id}
