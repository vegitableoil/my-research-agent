"""Experiment & Evaluation Layer — dataset prep, training, metrics, reporting."""

from __future__ import annotations

from typing import Any

from ids_agent.agents.base import AgentContext, BaseAgent


# ── Dataset Manager ───────────────────────────────────────────────────────────

KNOWN_DATASETS = [
    "NSL-KDD", "CICIDS2017", "CICIDS2018", "UNSW-NB15",
    "CIC-DDoS2019", "BOT-IoT", "UNSW-IoT", "KDD-Cup99",
]


class DatasetManagerAgent(BaseAgent):
    """
    Downloads, validates, and preprocesses a public IDS dataset.

    ctx.data inputs:
        dataset_name (str)           — one of KNOWN_DATASETS or a custom path
        dataset_config (dict)        — split ratios, balance strategy, features, …

    ctx.data outputs:
        dataset_artifact_path (str)
        feature_schema (dict)
        class_distribution (dict)
    """

    rag_indices = ["datasets_meta"]

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        name: str = ctx.data["dataset_name"]
        config: dict = ctx.data.get("dataset_config", {})

        meta = await self.retrieve(name, index="datasets_meta", k=3)

        raw_path: str = await self.tools.call("download_dataset", name=name, meta=meta)
        processed_path: str = await self.tools.call(
            "preprocess_dataset",
            raw_path=raw_path,
            config=config,
        )
        schema: dict = await self.tools.call("infer_feature_schema", path=processed_path)
        dist: dict = await self.tools.call("compute_class_distribution", path=processed_path)

        self.logger.info("Dataset '%s' ready → %s", name, processed_path)
        return {
            "dataset_artifact_path": processed_path,
            "feature_schema": schema,
            "class_distribution": dist,
        }


# ── Training Orchestrator ─────────────────────────────────────────────────────

class TrainingOrchestratorAgent(BaseAgent):
    """
    Runs training for the proposed model and all registered baselines.

    ctx.data inputs:
        proposed_model_path (str)     — path to generated model code
        baseline_registry (list[dict])
        dataset_artifact_path (str)
        hpo_config (dict, optional)   — if present, run HPO before final fit

    ctx.data outputs:
        run_manifest (list[dict])     — {model_name, checkpoint_path, config}
    """

    rag_indices = []

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        proposed_path: str = ctx.data["proposed_model_path"]
        baselines: list[dict] = ctx.data.get("baseline_registry", [])
        dataset_path: str = ctx.data["dataset_artifact_path"]
        hpo_config: dict | None = ctx.data.get("hpo_config")

        models = [{"name": "proposed", "path": proposed_path}] + baselines

        run_manifest: list[dict] = []
        for model in models:
            ckpt: str = await self.tools.call(
                "train_model",
                model_path=model["path"],
                dataset_path=dataset_path,
                hpo_config=hpo_config if model["name"] == "proposed" else None,
            )
            run_manifest.append({
                "model_name": model["name"],
                "checkpoint_path": ckpt,
            })
            self.logger.info("Trained model '%s' → %s", model["name"], ckpt)

        return {"run_manifest": run_manifest}


# ── Evaluation Agent ──────────────────────────────────────────────────────────

class EvaluationAgent(BaseAgent):
    """
    Computes the full IDS metric suite for each trained model.

    ctx.data inputs:
        run_manifest (list[dict])
        dataset_artifact_path (str)

    ctx.data outputs:
        evaluation_results (list[dict])   — per-model metric records
    """

    rag_indices = []

    IDS_METRICS = [
        "accuracy", "precision", "recall", "f1",
        "detection_rate", "false_alarm_rate",
        "roc_auc", "pr_auc",
        "per_class_f1", "inference_latency_ms", "model_size_mb",
    ]

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        manifest: list[dict] = ctx.data["run_manifest"]
        dataset_path: str = ctx.data["dataset_artifact_path"]

        evaluation_results: list[dict] = []
        for entry in manifest:
            metrics: dict = await self.tools.call(
                "evaluate_model",
                checkpoint_path=entry["checkpoint_path"],
                dataset_path=dataset_path,
                metric_names=self.IDS_METRICS,
            )
            stat_tests: dict = await self.tools.call(
                "run_statistical_tests",
                predictions_path=metrics.pop("predictions_path"),
            )
            evaluation_results.append({
                "model_name": entry["model_name"],
                **metrics,
                "statistical_tests": stat_tests,
            })
            self.logger.info("Evaluated '%s'", entry["model_name"])

        return {"evaluation_results": evaluation_results}


# ── Comparison Report Agent ───────────────────────────────────────────────────

class ComparisonReportAgent(BaseAgent):
    """
    Generates a human-readable comparison report with tables, plots, and narrative.

    ctx.data inputs:
        evaluation_results (list[dict])

    ctx.data outputs:
        report_md_path (str)
        report_pdf_path (str)
        plots_dir (str)
    """

    rag_indices = ["papers_global"]

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        results: list[dict] = ctx.data["evaluation_results"]

        # RAG: pull findings from literature to contextualise numbers
        sota_context = await self.retrieve(
            "state of the art IDS performance comparison", index="papers_global", k=8
        )

        table_md: str = await self.tools.call("generate_comparison_table", results=results)
        plots_dir: str = await self.tools.call(
            "generate_plots",
            results=results,
            plot_types=["roc_curves", "confusion_matrix", "learning_curves", "radar_chart"],
        )
        narrative: str = await self.tools.call(
            "generate_narrative",
            results=results,
            sota_context=sota_context,
            table=table_md,
        )

        md_path: str = await self.tools.call(
            "save_artifact",
            namespace="reports",
            name=f"comparison_{ctx.run_id}.md",
            content=f"{narrative}\n\n{table_md}",
        )
        pdf_path: str = await self.tools.call("render_pdf", md_path=md_path)

        self.logger.info("Comparison report → %s", md_path)
        return {
            "report_md_path": md_path,
            "report_pdf_path": pdf_path,
            "plots_dir": plots_dir,
        }
