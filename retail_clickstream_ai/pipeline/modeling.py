"""Model training, evaluation, and persistence — implemented in Stage 4.

Compares a current-category transition baseline, multinomial logistic
regression, and a random forest with fixed seeds. Selects the winner on
validation macro F1, evaluates it once on the held-out test month, and persists
the full preprocessing+estimator pipeline as ``model.joblib`` with metadata.

Determinism boundary: this module owns every number. Candidates are fit on the
**training** split only; hyperparameters and the winner are chosen on the
**validation** split; the locked winner is evaluated **once** on the **test**
split. ``run_candidate_experiments`` never receives test rows; only
``lock_winner_and_evaluate_test`` reads the test split, exactly once. The
Scientist crew's tools call these functions — agents interpret results, they
never compute a metric or select a model.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retail_clickstream_ai.crews.io_helpers import rel, write_json
from retail_clickstream_ai.pipeline import data as d
from retail_clickstream_ai.pipeline import features as F

# --------------------------------------------------------------------------- #
# Predeclared, immutable experiment configuration.
# --------------------------------------------------------------------------- #
SEED = 42
PRIMARY_METRIC = "macro_f1"
CLASSES: tuple[int, ...] = (1, 2, 3, 4)
# Display-only labels transcribed from the dataset description (NOT a constraint).
CATEGORY_LABELS: dict[int, str] = {1: "trousers", 2: "skirts", 3: "blouses", 4: "sale"}

BASELINE_ID = "baseline_transition"
LOGREG_ID = "logistic_regression"
RF_ID = "random_forest"

REQUIRED_FAMILIES: tuple[str, ...] = (
    "current_category_transition_baseline",
    "multinomial_logistic_regression",
    "random_forest",
)

# The tie-break is applied after the primary macro_f1 ranking (all descending
# except log_loss, then candidate_id ascending as the final deterministic key).
TIE_BREAK = ("weighted_f1_desc", "accuracy_desc", "log_loss_asc", "candidate_id_asc")

EXPERIMENT_CONFIG: dict[str, Any] = {
    "seed": SEED,
    "primary_metric": PRIMARY_METRIC,
    "secondary_metrics": ["accuracy", "weighted_f1", "log_loss", "per_class_precision_recall_f1"],
    "selection_rule": (
        "highest validation macro_f1; tie-break: weighted_f1 desc, accuracy desc, "
        "log_loss asc, candidate_id asc"
    ),
    "tie_break": list(TIE_BREAK),
    "classes": list(CLASSES),
    "preprocessing": {
        "categorical": "OneHotEncoder(handle_unknown='ignore')",
        "numeric_logreg": "StandardScaler",
        "numeric_rf": "passthrough",
    },
    "candidates": [
        {
            "candidate_id": BASELINE_ID,
            "family": "current_category_transition_baseline",
            "estimator": "CurrentCategoryTransitionBaseline",
            "hyperparameters": [{}],
        },
        {
            "candidate_id": LOGREG_ID,
            "family": "multinomial_logistic_regression",
            "estimator": "LogisticRegression",
            "hyperparameters": [{"C": 0.5}, {"C": 1.0}],
        },
        {
            "candidate_id": RF_ID,
            "family": "random_forest",
            "estimator": "RandomForestClassifier",
            "hyperparameters": [
                {"n_estimators": 200, "max_depth": None},
                {"n_estimators": 200, "max_depth": 20},
            ],
        },
    ],
}

_RELEVANT_PACKAGES = ("scikit-learn", "numpy", "pandas", "scipy", "joblib")
_ROUND = 6


def experiment_config() -> dict[str, Any]:
    """Return a deep copy of the immutable experiment configuration."""
    import copy

    return copy.deepcopy(EXPERIMENT_CONFIG)


def write_experiment_config(path: str | Path) -> Path:
    """Write ``experiment_config.json`` deterministically and return its path."""
    return write_json(Path(path), EXPERIMENT_CONFIG)


# --------------------------------------------------------------------------- #
# Current-category transition baseline (a real, probabilistic baseline).
# --------------------------------------------------------------------------- #
class CurrentCategoryTransitionBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the next category from the current one via learned transitions.

    Fits ``P(next | current_main_category)`` from the training transitions and
    predicts the arg-max next category; ``predict_proba`` returns the learned
    transition distribution (enabling log loss). An unseen current category at
    inference falls back to the training marginal distribution.
    """

    classes_: np.ndarray
    marginal_: np.ndarray
    transition_: dict[int, np.ndarray]

    def __init__(self, category_column: str = F.CURRENT_CATEGORY) -> None:
        self.category_column = category_column

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> CurrentCategoryTransitionBaseline:
        self.classes_ = np.array(CLASSES, dtype=int)
        y_arr = np.asarray(y, dtype=int)
        cur = np.asarray(X[self.category_column], dtype=int)
        n_classes = len(self.classes_)
        index = {c: i for i, c in enumerate(self.classes_)}
        # Marginal (fallback) distribution.
        marginal = np.zeros(n_classes, dtype=float)
        for c in y_arr:
            marginal[index[int(c)]] += 1.0
        self.marginal_ = marginal / max(marginal.sum(), 1.0)
        # Per-current-category transition distribution.
        table: dict[int, np.ndarray] = {}
        for c in self.classes_:
            mask = cur == c
            counts = np.zeros(n_classes, dtype=float)
            for nxt in y_arr[mask]:
                counts[index[int(nxt)]] += 1.0
            table[int(c)] = counts / counts.sum() if counts.sum() > 0 else self.marginal_
        self.transition_ = table
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cur = np.asarray(X[self.category_column], dtype=int)
        rows = [self.transition_.get(int(c), self.marginal_) for c in cur]
        return np.vstack(rows) if rows else np.empty((0, len(self.classes_)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# --------------------------------------------------------------------------- #
# Estimator + preprocessing construction
# --------------------------------------------------------------------------- #
def _make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_tf: Any = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(F.CATEGORICAL_FEATURES)),
            ("num", numeric_tf, list(F.NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )


def build_estimator(candidate_id: str, params: dict[str, Any], seed: int = SEED) -> Any:
    """Construct a fresh, unfitted estimator for a candidate + hyperparameters."""
    if candidate_id == BASELINE_ID:
        return CurrentCategoryTransitionBaseline()
    if candidate_id == LOGREG_ID:
        return Pipeline(
            [
                ("pre", _make_preprocessor(scale_numeric=True)),
                (
                    "clf",
                    LogisticRegression(
                        C=float(params["C"]),
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if candidate_id == RF_ID:
        return Pipeline(
            [
                ("pre", _make_preprocessor(scale_numeric=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown candidate_id: {candidate_id}")


def _Xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = frame.loc[:, list(F.PREDICTOR_COLUMNS)].copy()
    y = frame[F.TARGET].astype(int).to_numpy()
    return X, y


def _split_frames(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = features_df[features_df[F.SPLIT_COLUMN] == "train"]
    val = features_df[features_df[F.SPLIT_COLUMN] == "validation"]
    test = features_df[features_df[F.SPLIT_COLUMN] == "test"]
    return train, val, test


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, Any]:
    labels = list(CLASSES)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(c): {
            "precision": round(float(precision[i]), _ROUND),
            "recall": round(float(recall[i]), _ROUND),
            "f1": round(float(f1[i]), _ROUND),
            "support": int(support[i]),
        }
        for i, c in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    ll = (
        round(float(log_loss(y_true, y_proba, labels=labels)), _ROUND)
        if y_proba is not None
        else None
    )
    class_distribution = {str(c): int((y_true == c).sum()) for c in labels}
    return {
        "macro_f1": round(float(macro_f1), _ROUND),
        "weighted_f1": round(float(weighted_f1), _ROUND),
        "accuracy": round(float(acc), _ROUND),
        "log_loss": ll,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_distribution": class_distribution,
    }


def _rank_key(entry: dict[str, Any]) -> tuple[float, float, float, float, str]:
    """Deterministic sort key: macro_f1, then the predeclared tie-break."""
    m = entry["metrics"]
    ll = m["log_loss"] if m["log_loss"] is not None else float("inf")
    return (
        -m["macro_f1"],
        -m["weighted_f1"],
        -m["accuracy"],
        ll,
        entry["candidate_id"],
    )


# --------------------------------------------------------------------------- #
# Candidate training (train + validation only; NO test access)
# --------------------------------------------------------------------------- #
def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in _RELEVANT_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - all are installed
            packages[name] = "unknown"
    return {
        "python_version": platform.python_version(),
        "package_versions": packages,
        "seed": SEED,
    }


def run_candidate_experiments(
    features_df: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fit each candidate on train, tune on validation. Test is never touched."""
    cfg = config or EXPERIMENT_CONFIG
    seed = int(cfg["seed"])
    train, val, _test_withheld = _split_frames(features_df)
    X_tr, y_tr = _Xy(train)
    X_val, y_val = _Xy(val)

    candidates: list[dict[str, Any]] = []
    for spec in cfg["candidates"]:
        candidate_id = spec["candidate_id"]
        trials: list[dict[str, Any]] = []
        for params in spec["hyperparameters"]:
            est = build_estimator(candidate_id, params, seed)
            t0 = time.perf_counter()
            est.fit(X_tr, y_tr)
            train_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            y_pred = est.predict(X_val)
            inference_seconds = time.perf_counter() - t1
            y_proba = est.predict_proba(X_val) if hasattr(est, "predict_proba") else None
            metrics = _evaluate(y_val, np.asarray(y_pred), y_proba)
            trials.append(
                {
                    "params": params,
                    "metrics": metrics,
                    "train_seconds": round(train_seconds, 4),
                    "inference_seconds": round(inference_seconds, 4),
                }
            )
        # Best hyperparameter setting within this family, by validation macro_f1.
        best = min(
            trials,
            key=lambda t: (
                -t["metrics"]["macro_f1"],
                -t["metrics"]["weighted_f1"],
                -t["metrics"]["accuracy"],
            ),
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "family": spec["family"],
                "status": "PASS",
                "best_params": best["params"],
                "metrics": best["metrics"],
                "train_seconds": best["train_seconds"],
                "inference_seconds": best["inference_seconds"],
                "trials": trials,
            }
        )

    ranking = sorted(candidates, key=_rank_key)
    validation_ranking = [
        {
            "rank": i + 1,
            "candidate_id": c["candidate_id"],
            "macro_f1": c["metrics"]["macro_f1"],
            "evidence_ref": (
                f"candidate_results.json#candidates.{c['candidate_id']}.metrics.macro_f1"
            ),
        }
        for i, c in enumerate(ranking)
    ]

    present_families = {c["family"] for c in candidates}
    return {
        "primary_metric": PRIMARY_METRIC,
        "seed": seed,
        "classes": list(CLASSES),
        "candidate_count": len(candidates),
        "all_required_candidates_present": set(REQUIRED_FAMILIES) <= present_families,
        "candidates": {c["candidate_id"]: c for c in candidates},
        "validation_ranking": validation_ranking,
        "test_accessed": False,
        "environment": _environment(),
        "validation_rows": int(len(val)),
        "train_rows": int(len(train)),
    }


_REQUIRED_METRIC_KEYS = ("macro_f1", "weighted_f1", "accuracy", "per_class", "confusion_matrix")


def validate_training_outputs(candidate_results: dict[str, Any]) -> dict[str, Any]:
    """Check candidate completeness, metric schema, and that test was not accessed."""
    issues: list[str] = []
    candidates = candidate_results.get("candidates", {})
    if candidate_results.get("test_accessed") is not False:
        issues.append("test_accessed must be False during training")
    if len(candidates) < 3:
        issues.append(f"expected >=3 candidates, found {len(candidates)}")
    present_families = {c.get("family") for c in candidates.values()}
    if not set(REQUIRED_FAMILIES) <= present_families:
        issues.append(
            f"missing required families: {sorted(set(REQUIRED_FAMILIES) - present_families)}"
        )
    for cid, c in candidates.items():
        for key in _REQUIRED_METRIC_KEYS:
            if key not in c.get("metrics", {}):
                issues.append(f"candidate {cid} missing metric {key}")
    env = candidate_results.get("environment", {})
    if not env.get("package_versions") or "python_version" not in env:
        issues.append("reproducibility metadata (env/versions) missing")
    return {
        "passed": not issues,
        "test_accessed": candidate_results.get("test_accessed"),
        "candidate_count": len(candidates),
        "all_required_candidates_present": set(REQUIRED_FAMILIES) <= present_families,
        "issues": issues,
    }


# --------------------------------------------------------------------------- #
# Lock winner + one-time held-out test evaluation + persistence
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationBundle:
    """Everything the governance report + validators need after test eval."""

    selected_candidate_id: str
    model_path: Path
    metrics_path: Path
    metadata_path: Path
    figures: list[dict[str, Any]]
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    round_trip_validated: bool


def _confusion_figure(
    cm: list[list[int]], title: str, out_path: Path
) -> None:  # pragma: no cover - visual artifact
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = [CATEGORY_LABELS[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set(title=title, xlabel="Predicted next category", ylabel="True next category")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def lock_winner_and_evaluate_test(
    features_df: pd.DataFrame,
    scientist_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    candidate_results: dict[str, Any] | None = None,
    training_data_sha256: str | None = None,
    features_sha256: str | None = None,
    render_figures: bool = True,
) -> EvaluationBundle:
    """Select the winner on validation macro F1, evaluate once on test, persist."""
    cfg = config or EXPERIMENT_CONFIG
    seed = int(cfg["seed"])
    results = candidate_results or run_candidate_experiments(features_df, cfg)

    # Mechanical winner selection (no human/agent choice).
    winner_id = results["validation_ranking"][0]["candidate_id"]
    winner = results["candidates"][winner_id]
    winner_params = winner["best_params"]

    train, val, test = _split_frames(features_df)
    X_tr, y_tr = _Xy(train)
    X_test, y_test = _Xy(test)

    # Refit the winner on TRAIN ONLY (fit estimators only on training data).
    estimator = build_estimator(winner_id, winner_params, seed)
    t0 = time.perf_counter()
    estimator.fit(X_tr, y_tr)
    train_seconds = time.perf_counter() - t0

    # The one and only held-out test evaluation.
    t1 = time.perf_counter()
    test_pred = estimator.predict(X_test)
    inference_seconds = time.perf_counter() - t1
    test_proba = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    test_metrics = _evaluate(y_test, np.asarray(test_pred), test_proba)

    scientist_dir = Path(scientist_dir)
    scientist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Persist the complete preprocessing + estimator pipeline.
    model_path = scientist_dir / "model.joblib"
    joblib.dump(estimator, model_path)
    artifact_sha = d.sha256_file(model_path)

    # 2. Round-trip: reload and confirm prediction + probability parity.
    # Safe: this loads the trusted artifact we just wrote this instant (pickle
    # risk applies only to joblib files from untrusted sources; see model card).
    reloaded = joblib.load(model_path)
    rt_pred = reloaded.predict(X_test)
    predictions_match = bool(np.array_equal(np.asarray(rt_pred), np.asarray(test_pred)))
    proba_match = True
    if test_proba is not None:
        rt_proba = reloaded.predict_proba(X_test)
        proba_match = bool(np.allclose(rt_proba, test_proba, atol=1e-12))
    round_trip_validated = predictions_match and proba_match

    # 3. Figures (confusion matrices).
    figures: list[dict[str, Any]] = []
    if render_figures:
        fig_dir = scientist_dir / "figures"
        for split_name, m in (("validation", winner["metrics"]), ("test", test_metrics)):
            fig_path = fig_dir / f"confusion_matrix_{split_name}.png"
            _confusion_figure(
                m["confusion_matrix"],
                f"Confusion matrix — {split_name} ({winner_id})",
                fig_path,
            )
            figures.append(
                {
                    "figure_id": f"confusion_matrix_{split_name}",
                    "split": split_name,
                    "path": rel(fig_path),
                    "kind": "heatmap",
                    "evidence_ref": f"metrics.json#{split_name}.confusion_matrix",
                }
            )
        write_json(fig_dir / "figure_manifest.json", {"figures": figures})

    # 4. metrics.json (single source of truth for every reported number).
    from retail_clickstream_ai.validation.contract import DatasetContract

    class_labels = {str(c): CATEGORY_LABELS[c] for c in CLASSES}
    manifest = F.build_split_manifest(features_df, DatasetContract.build())
    metrics = {
        "selected_candidate_id": winner_id,
        "selected_model": winner["family"],
        "primary_metric": PRIMARY_METRIC,
        "seed": seed,
        "classes": list(CLASSES),
        "class_labels": class_labels,
        "selection_rule": cfg["selection_rule"],
        "split_counts": {
            "train_rows": manifest["train_rows"],
            "validation_rows": manifest["validation_rows"],
            "test_rows": manifest["test_rows"],
            "train_sessions": manifest["splits"]["train"]["sessions"],
            "validation_sessions": manifest["splits"]["validation"]["sessions"],
            "test_sessions": manifest["splits"]["test"]["sessions"],
        },
        "validation": winner["metrics"],
        "test": test_metrics,
        "candidates": {
            cid: {
                "family": c["family"],
                "best_params": c["best_params"],
                "macro_f1": c["metrics"]["macro_f1"],
                "weighted_f1": c["metrics"]["weighted_f1"],
                "accuracy": c["metrics"]["accuracy"],
                "log_loss": c["metrics"]["log_loss"],
            }
            for cid, c in results["candidates"].items()
        },
        "validation_ranking": results["validation_ranking"],
        "timing": {
            "winner_train_seconds": round(train_seconds, 4),
            "winner_inference_seconds": round(inference_seconds, 4),
        },
    }
    metrics_path = scientist_dir / "metrics.json"
    write_json(metrics_path, metrics)
    metrics_sha = d.sha256_file(metrics_path)

    # 5. model_metadata.json.
    dtypes = {c: str(X_tr[c].dtype) for c in F.PREDICTOR_COLUMNS}
    metadata = {
        "selected_model": winner["family"],
        "selected_candidate_id": winner_id,
        "hyperparameters": winner_params,
        "seed": seed,
        "target": F.TARGET,
        "input_schema": {
            "predictor_columns": list(F.PREDICTOR_COLUMNS),
            "categorical_features": list(F.CATEGORICAL_FEATURES),
            "numeric_features": list(F.NUMERIC_FEATURES),
            "dtypes": dtypes,
        },
        "feature_list": list(F.PREDICTOR_COLUMNS),
        "class_mapping": {
            "index_to_class": {str(i): int(c) for i, c in enumerate(CLASSES)},
            "class_to_label": class_labels,
        },
        "metrics_file": rel(metrics_path),
        "metrics_sha256": metrics_sha,
        "training_data_sha256": training_data_sha256,
        "features_sha256": features_sha256,
        "artifact_sha256": artifact_sha,
        "python_version": platform.python_version(),
        "package_versions": _environment()["package_versions"],
        "round_trip_validated": round_trip_validated,
        "security_note": (
            "model.joblib is pickle-based. Load ONLY the trusted, repo-produced, "
            "hash-verified artifact; never load a joblib file from an untrusted source."
        ),
    }
    metadata_path = scientist_dir / "model_metadata.json"
    write_json(metadata_path, metadata)

    return EvaluationBundle(
        selected_candidate_id=winner_id,
        model_path=model_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        figures=figures,
        metrics=metrics,
        metadata=metadata,
        round_trip_validated=round_trip_validated,
    )
