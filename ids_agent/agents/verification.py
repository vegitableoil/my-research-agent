"""
Verification Layer.

CodeVerifierAgent    — verifies architecture code produced by ArchitectureCodegenAgent.
BaselineVerifierAgent — verifies a baseline implementation against its source paper.

These are intentionally separate agents with no shared state so that verification
is always an independent, adversarial pass.
"""

from __future__ import annotations

from typing import Any

from ids_agent.agents.base import AgentContext, BaseAgent


class VerificationResult:
    def __init__(self, passed: bool, report: str, diff: str = "") -> None:
        self.passed = passed
        self.report = report
        self.diff = diff  # suggested corrections when failed

    def as_dict(self) -> dict:
        return {"passed": self.passed, "report": self.report, "diff": self.diff}


class CodeVerifierAgent(BaseAgent):
    """
    Independent code review agent for human-proposed architecture implementations.

    Checks:
      1. Static analysis  (pylint, mypy, bandit)
      2. Shape / dimension sanity  (dry-run with a small dummy batch)
      3. LLM-based code review  (correctness, edge cases, security, style)

    ctx.data inputs:
        generated_code (str)
        unit_tests (str)
        architecture_spec (str | dict)

    ctx.data outputs:
        verification_result (dict)      — passed, report, diff
    """

    rag_indices = ["papers_global", "code_snippets"]

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        code: str = ctx.data["generated_code"]
        tests: str = ctx.data.get("unit_tests", "")
        spec = ctx.data.get("architecture_spec", "")

        static_report: dict = await self.tools.call("run_static_analysis", code=code)
        shape_report: dict = await self.tools.call("run_shape_check", code=code)

        rag_context = await self.retrieve(str(spec), index="papers_global", k=5)
        llm_report: dict = await self.tools.call(
            "llm_code_review",
            code=code,
            tests=tests,
            spec=spec,
            rag_context=rag_context,
        )

        passed: bool = (
            static_report["ok"] and shape_report["ok"] and llm_report["ok"]
        )
        combined_report = "\n\n".join([
            static_report["text"],
            shape_report["text"],
            llm_report["text"],
        ])
        diff: str = llm_report.get("suggested_diff", "")

        result = VerificationResult(passed=passed, report=combined_report, diff=diff)
        self.logger.info("Code verification: passed=%s", passed)
        return {"verification_result": result.as_dict()}


class BaselineVerifierAgent(BaseAgent):
    """
    Verifies a baseline model implementation against the paper it is based on.

    Uses a *paper-specific* RAG index (paper_{paper_id}) so it has direct access
    to the source paper's architecture and reported results — but NO memory of
    the replication session, preserving independence.

    ctx.data inputs:
        baseline_code (str)
        paper_id (str)              — used to select the per-paper RAG index
        reported_metrics (dict)     — from the paper summary

    ctx.data outputs:
        baseline_verification_result (dict)   — passed, report, diff
    """

    rag_indices = []  # loaded dynamically per paper_id

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        code: str = ctx.data["baseline_code"]
        paper_id: str = ctx.data["paper_id"]
        reported_metrics: dict = ctx.data.get("reported_metrics", {})

        index_name = f"paper_{paper_id}"

        arch_context = await self.retrieve(
            "model architecture layers dimensions", index=index_name, k=8
        )
        metric_context = await self.retrieve(
            "performance evaluation results table", index=index_name, k=4
        )

        arch_report: dict = await self.tools.call(
            "check_architecture_fidelity",
            code=code,
            paper_context=arch_context,
        )
        metric_report: dict = await self.tools.call(
            "check_metric_sanity",
            code=code,
            reported_metrics=reported_metrics,
            paper_context=metric_context,
        )

        passed: bool = arch_report["ok"] and metric_report["ok"]
        combined_report = "\n\n".join([arch_report["text"], metric_report["text"]])
        diff: str = arch_report.get("suggested_diff", "") + metric_report.get("suggested_diff", "")

        result = VerificationResult(passed=passed, report=combined_report, diff=diff)
        self.logger.info("Baseline verification for paper %s: passed=%s", paper_id, passed)
        return {"baseline_verification_result": result.as_dict()}
