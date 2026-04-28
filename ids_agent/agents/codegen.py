"""Model Development Layer — architecture codegen and HPO script generation."""

from __future__ import annotations

from typing import Any

from ids_agent.agents.base import AgentContext, BaseAgent


class ArchitectureCodegenAgent(BaseAgent):
    """
    Translates a human-supplied architecture specification into runnable code.

    ctx.data inputs:
        architecture_spec (str | dict)  — free-text or structured description
        framework (str)                 — "pytorch" | "tensorflow" | "sklearn"

    ctx.data outputs:
        generated_code (str)            — Python source
        unit_tests (str)                — Pytest source
        code_artifact_path (str)
    """

    rag_indices = ["papers_global", "code_snippets"]

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        spec = ctx.data["architecture_spec"]
        framework: str = ctx.data.get("framework", "pytorch")

        rag_context = await self.retrieve(str(spec), index="papers_global", k=6)
        code_examples = await self.retrieve(str(spec), index="code_snippets", k=4)

        generated_code: str = await self.tools.call(
            "generate_model_code",
            spec=spec,
            framework=framework,
            rag_context=rag_context,
            code_examples=code_examples,
        )
        unit_tests: str = await self.tools.call(
            "generate_unit_tests",
            code=generated_code,
            framework=framework,
        )

        artifact_path: str = await self.tools.call(
            "save_artifact",
            namespace="code",
            name=f"model_{ctx.run_id}.py",
            content=generated_code,
        )
        await self.tools.call(
            "save_artifact",
            namespace="code",
            name=f"test_model_{ctx.run_id}.py",
            content=unit_tests,
        )

        self.logger.info("Architecture code generated → %s", artifact_path)
        return {
            "generated_code": generated_code,
            "unit_tests": unit_tests,
            "code_artifact_path": artifact_path,
        }


class HyperparamTunerAgent(BaseAgent):
    """
    Writes HPO scripts (Optuna / Ray Tune) tailored to the generated architecture.

    ctx.data inputs:
        generated_code (str)         — model source
        architecture_spec (str)      — for extracting tunable parameters

    ctx.data outputs:
        hpo_config (dict)            — search space definition
        hpo_script_path (str)        — path to saved sweep script
    """

    rag_indices = ["papers_global"]

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        code: str = ctx.data["generated_code"]
        spec = ctx.data.get("architecture_spec", "")

        # Pull any HPO hints from related papers
        hints = await self.retrieve(
            f"hyperparameter tuning {spec}", index="papers_global", k=4
        )

        hpo_config: dict = await self.tools.call(
            "build_hpo_search_space",
            code=code,
            paper_hints=hints,
        )
        hpo_script: str = await self.tools.call(
            "generate_hpo_script",
            hpo_config=hpo_config,
            framework="optuna",   # configurable
        )
        script_path: str = await self.tools.call(
            "save_artifact",
            namespace="hpo",
            name=f"hpo_sweep_{ctx.run_id}.py",
            content=hpo_script,
        )

        self.logger.info("HPO script written → %s", script_path)
        return {"hpo_config": hpo_config, "hpo_script_path": script_path}
