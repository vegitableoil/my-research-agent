"""Literature Intelligence Layer — search, summarise, and index papers."""

from __future__ import annotations

from typing import Any

from ids_agent.agents.base import AgentContext, BaseAgent
from ids_agent.human_loop.gate import Decision, HumanGate


class LiteratureSearchAgent(BaseAgent):
    """
    Queries multiple academic sources and returns a ranked list of candidate papers.

    ctx.data inputs:
        search_query (str)             — keywords / topic
        filters (dict, optional)       — year_range, venues, min_citations, …

    ctx.data outputs:
        candidate_papers (list[dict])  — id, title, abstract, url, source, score
    """

    rag_indices = []  # no RAG needed for raw search

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        query: str = ctx.data["search_query"]
        filters: dict = ctx.data.get("filters", {})

        results = await self.tools.call("search_papers", query=query, filters=filters)
        deduplicated = await self.tools.call("dedup_and_rank", papers=results)

        self.logger.info("Found %d candidate papers for query: %s", len(deduplicated), query)
        return {"candidate_papers": deduplicated}


class SummaryGeneratorAgent(BaseAgent):
    """
    Generates a structured summary for a single approved paper.

    ctx.data inputs:
        approved_paper (dict)   — paper record including full-text path or URL

    ctx.data outputs:
        summary (dict)          — novelty, system_design, datasets, architecture,
                                  performance_metrics, limitations, implications,
                                  citation_graph
    """

    rag_indices = ["papers_global"]

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        paper = ctx.data["approved_paper"]
        full_text: str = await self.tools.call("fetch_full_text", paper_id=paper["id"])

        # RAG: pull related already-indexed papers for comparative context
        related = await self.retrieve(paper["title"], index="papers_global", k=5)

        summary: dict = await self.tools.call(
            "extract_structured_summary",
            text=full_text,
            related_context=related,
        )
        summary["paper_id"] = paper["id"]
        self.logger.info("Summary generated for paper %s", paper["id"])
        return {"summary": summary}


class LiteraturePipeline:
    """
    Orchestrates the search → human-review → summarise loop.

    Intended to be called by the top-level OrchestratorAgent.
    """

    def __init__(
        self,
        search_agent: LiteratureSearchAgent,
        summary_agent: SummaryGeneratorAgent,
        human_gate: HumanGate,
    ) -> None:
        self.search_agent = search_agent
        self.summary_agent = summary_agent
        self.human_gate = human_gate

    async def run_loop(self, ctx: AgentContext) -> list[dict]:
        """Return list of summaries for all human-approved papers."""
        summaries: list[dict] = []

        search_result = await self.search_agent.run(ctx)
        ctx.data.update(search_result)

        for paper in ctx.data["candidate_papers"]:
            decision: Decision = await self.human_gate.request(
                prompt=f"Approve paper?\n\nTitle: {paper['title']}\n\nAbstract: {paper.get('abstract', '')}",
                options=["approve", "reject", "stop"],
                paper=paper,
            )
            if decision.choice == "stop":
                break
            if decision.choice == "reject":
                continue

            ctx.data["approved_paper"] = paper
            result = await self.summary_agent.run(ctx)
            summaries.append(result["summary"])

        return summaries
