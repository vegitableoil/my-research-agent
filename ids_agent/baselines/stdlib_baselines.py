"""
Pre-built adapters for standard ML baselines (no paper replication needed).

Each adapter wraps a sklearn-compatible estimator behind a unified interface
so the TrainingOrchestratorAgent can treat them identically to paper models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BaselineSpec:
    name: str
    model_type: str           # "stdlib"
    artifact_path: str = ""   # set after fit


class StdlibBaselineAdapter:
    """Thin wrapper that exposes fit / predict / save for any sklearn estimator."""

    def __init__(self, estimator: Any, name: str) -> None:
        self.estimator = estimator
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StdlibBaselineAdapter":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        decision = self.estimator.decision_function(X)
        # sigmoid for binary; softmax otherwise handled upstream
        return 1 / (1 + np.exp(-decision))

    def save(self, path: str) -> str:
        import joblib
        joblib.dump(self.estimator, path)
        return path

    @classmethod
    def load(cls, path: str, name: str) -> "StdlibBaselineAdapter":
        import joblib
        estimator = joblib.load(path)
        return cls(estimator=estimator, name=name)


# ── Factory functions ──────────────────────────────────────────────────────────

def random_forest(n_estimators: int = 100, **kwargs: Any) -> StdlibBaselineAdapter:
    from sklearn.ensemble import RandomForestClassifier
    return StdlibBaselineAdapter(
        estimator=RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, **kwargs),
        name="RandomForest",
    )


def mlp(hidden_layer_sizes: tuple = (256, 128), **kwargs: Any) -> StdlibBaselineAdapter:
    from sklearn.neural_network import MLPClassifier
    return StdlibBaselineAdapter(
        estimator=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, **kwargs),
        name="MLP",
    )


def svm(**kwargs: Any) -> StdlibBaselineAdapter:
    from sklearn.svm import SVC
    return StdlibBaselineAdapter(
        estimator=SVC(probability=True, **kwargs),
        name="SVM",
    )


def xgboost(**kwargs: Any) -> StdlibBaselineAdapter:
    from xgboost import XGBClassifier
    return StdlibBaselineAdapter(
        estimator=XGBClassifier(eval_metric="logloss", **kwargs),
        name="XGBoost",
    )


def decision_tree(**kwargs: Any) -> StdlibBaselineAdapter:
    from sklearn.tree import DecisionTreeClassifier
    return StdlibBaselineAdapter(
        estimator=DecisionTreeClassifier(**kwargs),
        name="DecisionTree",
    )


# Registry for easy lookup by name
STDLIB_FACTORIES: dict[str, Any] = {
    "random_forest": random_forest,
    "mlp": mlp,
    "svm": svm,
    "xgboost": xgboost,
    "decision_tree": decision_tree,
}


def build_stdlib_baseline(name: str, **kwargs: Any) -> StdlibBaselineAdapter:
    factory = STDLIB_FACTORIES.get(name.lower().replace(" ", "_"))
    if factory is None:
        raise ValueError(
            f"Unknown stdlib baseline '{name}'. Available: {list(STDLIB_FACTORIES)}"
        )
    return factory(**kwargs)
