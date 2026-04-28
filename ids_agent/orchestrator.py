"""
OrchestratorAgent — top-level pipeline state machine.

Coordinates all agent clusters, manages human-in-the-loop gates,
and persists pipeline state so runs can be resumed after interruption.
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum, auto
from typing import Any

from ids_agent.agents.base import AgentContext
from ids_agent.agents.baseline import BaselinePipeline
from ids_agent.agents.codegen import ArchitectureCodegenAgent, HyperparamTunerAgent
from ids_agent.agents.evaluation import (
    ComparisonReportAgent,
    DatasetManagerAgent,
    EvaluationAgent,
    TrainingOrchestratorAgent,
)
from ids_agent.agents.literature import LiteraturePipeline
from ids_agent.agents.rag_indexing import RAGIndexingAgent
from ids_agent.agents.verification import CodeVerifierAgent
from ids_agent.human_loop.gate import HumanGate

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = auto()
    LITERATURE_SEARCH = auto()
    HUMAN_PAPER_REVIEW = auto()
    SUMMARISING = auto()
    RAG_INDEXING = auto()
    AWAITING_ARCH_SPEC = auto()
    CODE_GENERATION = auto()
    CODE_VERIFICATION = auto()
    HPO_SCRIPT_GEN = auto()
    AWAITING_BASELINE = auto()
    BASELINE_LOCATE = auto()
    BASELINE_VERIFY = auto()
    AWAITING_DATASET = auto()
    DATA_PREP = auto()
    TRAINING = auto()
    EVALUATION = auto()
    REPORTING = auto()
    DONE = auto()
    FAILED = auto()


class AgentRegistry:
    """Global registry mapping string names → agent classes."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, agent_cls: type) -> None:
        cls._registry[name] = agent_cls

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(f"Agent '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())


class OrchestratorAgent:
    """
    Drives the end-to-end ML-IDS pipeline.

    Instantiate once, then call the appropriate `run_*` methods in sequence
    or call `run_full_pipeline()` for an automated end-to-end execution.
    """

    def __init__(
        self,
        literature_pipeline: LiteraturePipeline,
        rag_indexing_agent: RAGIndexingAgent,
        codegen_agent: ArchitectureCodegenAgent,
        code_verifier: CodeVerifierAgent,
        hpo_agent: HyperparamTunerAgent,
        baseline_pipeline: BaselinePipeline,
        dataset_agent: DatasetManagerAgent,
        training_agent: TrainingOrchestratorAgent,
        evaluation_agent: EvaluationAgent,
        report_agent: ComparisonReportAgent,
        human_gate: HumanGate,
        max_codegen_retries: int = 3,
    ) -> None:
        self.lit = literature_pipeline
        self.rag_idx = rag_indexing_agent
        self.codegen = codegen_agent
        self.verifier = code_verifier
        self.hpo = hpo_agent
        self.baseline = baseline_pipeline
        self.dataset = dataset_agent
        self.training = training_agent
        self.evaluation = evaluation_agent
        self.report = report_agent
        self.gate = human_gate
        self.max_codegen_retries = max_codegen_retries
        self.state = PipelineState.IDLE

    # ── Public entry points ────────────────────────────────────────────────────

    async def run_literature_phase(self, search_query: str, filters: dict | None = None) -> list[dict]:
        """Run the search → review → summarise → index loop."""
        ctx = self._new_ctx(search_query=search_query, filters=filters or {})
        self._transition(PipelineState.LITERATURE_SEARCH)

        summaries = await self.lit.run_loop(ctx)
        for summary in summaries:
            ctx.data.update({"summary": summary})
            self._transition(PipelineState.RAG_INDEXING)
            await self.rag_idx.run(ctx)

        self._transition(PipelineState.IDLE)
        return summaries

    async def run_development_phase(self, architecture_spec: Any, framework: str = "pytorch") -> dict:
        """Generate code for a human-proposed architecture, verify, then produce HPO scripts."""
        ctx = self._new_ctx(architecture_spec=architecture_spec, framework=framework)
        self._transition(PipelineState.CODE_GENERATION)

        for attempt in range(1, self.max_codegen_retries + 1):
            ctx.data.update(await self.codegen.run(ctx))
            self._transition(PipelineState.CODE_VERIFICATION)
            ctx.data.update(await self.verifier.run(ctx))

            if ctx.data["verification_result"]["passed"]:
                break

            if attempt < self.max_codegen_retries:
                ctx.data["correction_diff"] = ctx.data["verification_result"]["diff"]
                self._transition(PipelineState.CODE_GENERATION)
            else:
                await self.gate.request(
                    prompt=(
                        f"Code verification failed after {self.max_codegen_retries} attempts.\n"
                        f"Report:\n{ctx.data['verification_result']['report']}"
                    ),
                    options=["override_accept", "abort"],
                )

        self._transition(PipelineState.HPO_SCRIPT_GEN)
        ctx.data.update(await self.hpo.run(ctx))
        self._transition(PipelineState.IDLE)
        return ctx.data

    async def run_baseline_phase(self, baseline_choices: list[dict]) -> list[dict]:
        """Locate / replicate and verify each human-selected baseline."""
        results = []
        for choice in baseline_choices:
            ctx = self._new_ctx(**choice)
            self._transition(PipelineState.BASELINE_LOCATE)
            result = await self.baseline.run(ctx)
            results.append(result)
        self._transition(PipelineState.IDLE)
        return results

    async def run_evaluation_phase(
        self,
        proposed_model_path: str,
        baseline_registry: list[dict],
        dataset_name: str,
        dataset_config: dict | None = None,
    ) -> dict:
        """Prepare data, train all models, evaluate, and generate the comparison report."""
        ctx = self._new_ctx(
            proposed_model_path=proposed_model_path,
            baseline_registry=baseline_registry,
            dataset_name=dataset_name,
            dataset_config=dataset_config or {},
        )

        self._transition(PipelineState.DATA_PREP)
        ctx.data.update(await self.dataset.run(ctx))

        self._transition(PipelineState.TRAINING)
        ctx.data.update(await self.training.run(ctx))

        self._transition(PipelineState.EVALUATION)
        ctx.data.update(await self.evaluation.run(ctx))

        self._transition(PipelineState.REPORTING)
        ctx.data.update(await self.report.run(ctx))

        self._transition(PipelineState.DONE)
        return ctx.data

    # ── Private helpers ────────────────────────────────────────────────────────

    def _new_ctx(self, **kwargs: Any) -> AgentContext:
        return AgentContext(run_id=str(uuid.uuid4()), data=kwargs)

    def _transition(self, new_state: PipelineState) -> None:
        logger.info("Pipeline state: %s → %s", self.state.name, new_state.name)
        self.state = new_state
