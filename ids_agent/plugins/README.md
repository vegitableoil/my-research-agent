# Plugin System

Drop any Python file here that registers new agents or tools via the decorators:

```python
from ids_agent.tools.registry import tool, register_agent
```

Plugins are auto-discovered at startup — no changes to core code required.

## Planned plugins (future work)

| File | Description |
|------|-------------|
| `openml_connector.py` | Pull datasets from OpenML |
| `mlflow_tracker.py` | MLflow experiment tracking |
| `adversarial_eval.py` | FGSM / PGD robustness evaluation |
| `explainability.py` | SHAP / LIME feature importance reporter |
| `federated.py` | Federated learning coordinator |
