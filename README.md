# ML-IDS Research Agent

An end-to-end multi-agent system that automates the full ML-based Intrusion Detection System
research pipeline — from literature survey through experiment, comparison, and paper writing.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

---

## End-to-End Workflow

The pipeline has four sequential phases.  The agents handle all mechanical work;
you make every strategic decision.

```
Phase 1 → Collect Literature
Phase 2 → Implement & Tune Your Model
Phase 3 → Run Experiments & Compare Baselines
Phase 4 → Write the Paper
```

---

## Phase 1 — Collect Literature

**Goal**: build a curated knowledge base of relevant ML-IDS papers.

### 1.1  Start a literature search

```python
from ids_agent import OrchestratorAgent
from ids_agent.config import settings

orchestrator = OrchestratorAgent.from_settings(settings)

summaries = await orchestrator.run_literature_phase(
    search_query="deep learning intrusion detection system network traffic",
    filters={
        "year_range": [2019, 2025],
        "venues": ["IEEE S&P", "USENIX Security", "NDSS", "CCS", "TDSC", "TIFS"],
        "min_citations": 10,
    },
)
```

### 1.2  Review each paper (human gate)

The agent presents each candidate paper in the terminal:

```
╔══════════════════════════════════════════════════════╗
║              Human Review Required                   ║
║  Title: "Efficient CNN-based IDS for IoT Networks"   ║
║  Venue: IEEE TIFS 2023  |  Citations: 142            ║
║  Abstract: ...                                       ║
╚══════════════════════════════════════════════════════╝
Choose [approve] / [reject] / [stop]:
```

- **approve** — the agent generates a structured summary and indexes the paper into the RAG store.
- **reject** — skipped, moves to the next candidate.
- **stop** — ends the search loop with papers collected so far.

### 1.3  What gets stored automatically

For every approved paper the system saves to the database:

| Field | Description |
|-------|-------------|
| Novelty | What is new vs. prior work |
| System Design | Architecture, components, data flow |
| Datasets used | Which IDS datasets + preprocessing choices |
| Performance metrics | All reported numbers (accuracy, DR, FAR, F1, …) |
| Limitations | Stated and implied weaknesses |
| Implications | What this means for your research |

All full texts are chunked and embedded into the vector store for RAG retrieval in later phases.

---

## Phase 2 — Implement & Tune Your Model

**Goal**: generate verified, runnable code for the architecture you have designed.

### 2.1  Provide your architecture specification

```python
spec = """
Bidirectional LSTM with attention mechanism.
Input: 78-dimensional flow features (normalised).
Two BiLSTM layers (hidden=128, dropout=0.3),
followed by a self-attention layer,
then a fully-connected head for multi-class classification
(normal + 9 attack categories).
"""

result = await orchestrator.run_development_phase(
    architecture_spec=spec,
    framework="pytorch",          # "pytorch" | "tensorflow" | "sklearn"
)
```

### 2.2  Automatic code generation → verification loop

```
ArchitectureCodegenAgent  →  generates model code + unit tests
        ↓
CodeVerifierAgent         →  static analysis + shape check + LLM review
        ↓ (fail)
ArchitectureCodegenAgent  →  corrects based on diff  (up to 3 retries)
        ↓ (pass)
HyperparamTunerAgent      →  writes Optuna sweep script
```

On success, two artefacts are saved to `data/artifacts/code/` and `data/artifacts/hpo/`:

```
data/artifacts/
  code/
    model_<run_id>.py       ← your architecture, ready to train
    test_model_<run_id>.py  ← pytest suite
  hpo/
    hpo_sweep_<run_id>.py   ← Optuna search space + trial loop
```

### 2.3  Run hyperparameter search (optional but recommended)

```bash
python data/artifacts/hpo/hpo_sweep_<run_id>.py \
    --n-trials 100 \
    --dataset data/artifacts/datasets/CICIDS2017_processed/
```

---

## Phase 3 — Run Experiments & Compare Baselines

**Goal**: train your model and all baselines on the same dataset; produce a comparison table.

### 3.1  Select baselines

```python
baseline_choices = [
    # Standard models — no replication needed
    {"baseline_paper": None, "model_type": "stdlib", "name": "random_forest"},
    {"baseline_paper": None, "model_type": "stdlib", "name": "mlp"},
    {"baseline_paper": None, "model_type": "stdlib", "name": "xgboost"},

    # Paper-based model — agent searches GitHub; replicates if not found
    {
        "model_type": "paper",
        "paper_id": "doe2022efficient",        # must already be in the literature DB
        "baseline_paper": {
            "title": "Efficient CNN-based IDS for IoT Networks",
            "authors": ["Doe", "Smith"],
            "year": 2022,
        },
        "framework": "pytorch",
    },
]

baseline_results = await orchestrator.run_baseline_phase(baseline_choices)
```

For each paper-based baseline the agent:
1. Searches GitHub for an existing implementation (checks stars, README, last commit).
2. If none found — replicates the model from the paper using RAG over the paper's full text.
3. Runs `BaselineVerifierAgent` independently (it has no memory of replication) to check architecture fidelity and metric sanity against the paper.

### 3.2  Select dataset and run all experiments

```python
report = await orchestrator.run_evaluation_phase(
    proposed_model_path="data/artifacts/code/model_<run_id>.py",
    baseline_registry=baseline_results,
    dataset_name="CICIDS2017",                # see full list below
    dataset_config={
        "split": [0.7, 0.15, 0.15],          # train / val / test
        "balance_strategy": "smote",          # none | smote | undersample
        "feature_selection": "all",           # all | top_k | mutual_info
        "random_seed": 42,
    },
)
```

**Supported public datasets (built-in)**

| Dataset | Traffic type | Classes |
|---------|-------------|---------|
| `NSL-KDD` | Network flows | 5 (normal + 4 attack) |
| `CICIDS2017` | Network flows | 15 |
| `CICIDS2018` | Network flows | 15 |
| `UNSW-NB15` | Network flows | 10 |
| `CIC-DDoS2019` | DDoS traffic | 13 |
| `BOT-IoT` | IoT traffic | 5 |

### 3.3  Results

The evaluation agent computes the full IDS metric suite and saves outputs to
`data/artifacts/reports/` and `data/artifacts/plots/`:

```
data/artifacts/
  reports/
    comparison_<run_id>.md    ← Markdown table + LLM narrative
    comparison_<run_id>.pdf   ← rendered PDF
  plots/
    roc_curves.png
    confusion_matrix_<model>.png
    learning_curves.png
    radar_chart.png
```

Metrics computed per model: Accuracy, Precision, Recall, F1, Detection Rate, False Alarm Rate,
ROC-AUC, PR-AUC, per-class F1, inference latency (ms), model size (MB).
Statistical significance is tested with McNemar's and Wilcoxon signed-rank tests.

---

## Phase 4 — Write the Paper

**Goal**: produce a structured draft grounded in your results and the literature database.

> Paper writing is currently a **human-driven** phase supported by the RAG store.
> A `PaperWriterAgent` is on the roadmap (see `agents/paper_writer.py` in future modules).

### Use the RAG store as your research assistant

```python
from ids_agent.rag.retriever import RAGRetriever

retriever = RAGRetriever.from_settings(settings)

# Find how prior work reports detection rate on CICIDS2017
hits = await retriever.retrieve(
    query="detection rate false alarm rate CICIDS2017 deep learning",
    index="papers_global",
    k=8,
)

# Pull the methodology section of a specific paper
hits = await retriever.retrieve(
    query="preprocessing feature normalisation flow statistics",
    index="paper_doe2022efficient",
    k=5,
)
```

### Recommended paper structure

Use the summaries stored in the database (`data/ids_agent.db`, table `summaries`) to write each section:

| Section | Source material |
|---------|----------------|
| **Introduction** | `novelty` + `implications` fields of top-cited papers |
| **Related Work** | all `summary` records, grouped by approach (CNN / RNN / GNN / ensemble) |
| **Methodology** | your `architecture_spec` + generated `model_<run_id>.py` |
| **Experiments** | `dataset_config` + `comparison_<run_id>.md` |
| **Results** | plots in `data/artifacts/plots/` + statistical test output |
| **Discussion** | `limitations` fields + your model's failure modes from confusion matrix |
| **Conclusion** | `implications` fields + future work from `ARCHITECTURE.md` roadmap |

---

## Quick-Start Summary

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env          # fill in API keys (Anthropic, Semantic Scholar, GitHub)

# 3. Run the full pipeline
python -m ids_agent.cli \
  --query "deep learning intrusion detection" \
  --arch-spec specs/my_model.txt \
  --baselines random_forest mlp xgboost doe2022efficient \
  --dataset CICIDS2017
```

Or drive each phase interactively from a notebook or REPL using the
`OrchestratorAgent` API shown above.

---

## Project Structure

```
ids_agent/          Python package (agents, RAG, tools, DB, human-loop)
ARCHITECTURE.md     Full system design with Mermaid diagrams
pyproject.toml      Dependencies and build config
data/               Runtime data (created on first run, git-ignored)
  chroma/           Vector store
  ids_agent.db      Relational store
  artifacts/        Generated code, weights, plots, reports
```
