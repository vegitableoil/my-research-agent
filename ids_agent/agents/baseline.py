"""Baseline Pipeline — locate, replicate, and register baseline models."""

from __future__ import annotations

from typing import Any

from ids_agent.agents.base import AgentContext, BaseAgent
from ids_agent.agents.verification import BaselineVerifierAgent
from ids_agent.human_loop.gate import HumanGate


class BaselineLocatorAgent(BaseAgent):
    """
    Searches GitHub for an existing implementation of a paper-based baseline.

    ctx.data inputs:
        baseline_paper (dict)      — title, authors, year, abstract

    ctx.data outputs:
        github_repo (dict | None)  — url, stars, last_commit, quality_score
                                     None if no acceptable repo found
    """

    rag_indices = []

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        paper: dict = ctx.data["baseline_paper"]
        repos: list[dict] = await self.tools.call(
            "search_github_code",
            query=paper["title"],
            authors=paper.get("authors", []),
            keywords=paper.get("keywords", []),
        )
        best: dict | None = await self.tools.call(
            "rank_and_qualify_repos",
            repos=repos,
            min_quality_score=0.6,
        )
        self.logger.info(
            "GitHub search for '%s': %s",
            paper["title"],
            f"found {best['url']}" if best else "not found",
        )
        return {"github_repo": best}


class BaselineReplicatorAgent(BaseAgent):
    """
    Recreates a baseline model from scratch using the paper as ground truth.

    ctx.data inputs:
        baseline_paper (dict)   — paper record (must already be in RAG index)
        paper_id (str)
        framework (str)

    ctx.data outputs:
        baseline_code (str)
        baseline_artifact_path (str)
    """

    rag_indices = []  # loads per-paper index dynamically

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        paper: dict = ctx.data["baseline_paper"]
        paper_id: str = ctx.data["paper_id"]
        framework: str = ctx.data.get("framework", "pytorch")

        index_name = f"paper_{paper_id}"
        arch_context = await self.retrieve(
            "model architecture implementation details", index=index_name, k=10
        )

        code: str = await self.tools.call(
            "replicate_model_from_paper",
            paper=paper,
            arch_context=arch_context,
            framework=framework,
        )
        artifact_path: str = await self.tools.call(
            "save_artifact",
            namespace="code",
            name=f"baseline_{paper_id}_{ctx.run_id}.py",
            content=code,
        )
        self.logger.info("Replicated baseline for paper %s → %s", paper_id, artifact_path)
        return {"baseline_code": code, "baseline_artifact_path": artifact_path}


class BaselinePipeline:
    """
    Orchestrates locate → (replicate if missing) → verify loop for a single baseline.
    """

    def __init__(
        self,
        locator: BaselineLocatorAgent,
        replicator: BaselineReplicatorAgent,
        verifier: BaselineVerifierAgent,
        human_gate: HumanGate,
        max_retries: int = 3,
    ) -> None:
        self.locator = locator
        self.replicator = replicator
        self.verifier = verifier
        self.human_gate = human_gate
        self.max_retries = max_retries

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        # Try to find an existing implementation
        locate_result = await self.locator.run(ctx)
        ctx.data.update(locate_result)

        if ctx.data["github_repo"]:
            # Use the found repo's code
            ctx.data["baseline_code"] = await _fetch_repo_code(
                self.locator.tools, ctx.data["github_repo"]
            )
        else:
            ctx.data.update(await self.replicator.run(ctx))

        # Verify with retries
        for attempt in range(1, self.max_retries + 1):
            verify_result = await self.verifier.run(ctx)
            ctx.data.update(verify_result)

            if ctx.data["baseline_verification_result"]["passed"]:
                return ctx.data["baseline_verification_result"]

            if attempt < self.max_retries:
                # Feed the diff back to the replicator for correction
                ctx.data["correction_diff"] = ctx.data["baseline_verification_result"]["diff"]
                ctx.data.update(await self.replicator.run(ctx))
            else:
                # Human override gate
                await self.human_gate.request(
                    prompt=(
                        f"Baseline replication failed after {self.max_retries} attempts.\n"
                        f"Report:\n{ctx.data['baseline_verification_result']['report']}\n"
                        "Please review the diff and decide."
                    ),
                    options=["override_accept", "abort"],
                )

        return ctx.data["baseline_verification_result"]


async def _fetch_repo_code(tools: Any, repo: dict) -> str:
    return await tools.call("fetch_github_repo_code", repo_url=repo["url"])
