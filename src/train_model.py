"""
train_model.py

Trains a recovery-probability classifier on data/logged_transactions.csv
ONLY, estimating:

    P(recovered = 1 | transaction features, action)

`action` is included as a categorical feature so a single model can score
every candidate action for a transaction (see evaluate_policy.py).

This script never opens data/holdout_full_matrix.csv. Model selection is
based purely on cross-validated metrics on the logged data.

Run: python train_model.py   (from the src/ directory)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).parent))
from features import (
    BOOLEAN_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    to_model_matrix,
)

SEED = 42
N_SPLITS = 5

DATA_PATH = Path(__file__).parent.parent / "data" / "logged_transactions.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"


def load_training_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"action_taken": "action"})
    X = to_model_matrix(df)
    y = df["recovered"].to_numpy()
    return df, X, y


def evaluate_candidate(name, estimator, X, y, cv):
    """5-fold out-of-fold predictions -> CV metrics. No holdout involved."""
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
    oof_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y, oof_proba)),
        "log_loss": float(log_loss(y, oof_proba)),
        "brier": float(brier_score_loss(y, oof_proba)),
    }
    return metrics, oof_proba


def run_ips_diagnostic(df, X, y, cv, unweighted_auc):
    """
    OPTIONAL experiment (not the primary model).

    Estimates P(action | context features) from the logged data and checks
    whether inverse-propensity weighting changes cross-validated recovery
    prediction performance. Reported honestly, including if the propensity
    estimates look unreliable due to sparse action counts (e.g. payment_link
    n=39, escalate n=21 out of 800). This never changes the saved model.
    """
    context_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + ["failure_reason"]
    Xc = df[context_cols].copy()
    Xc["active_promise_to_pay"] = Xc["active_promise_to_pay"].astype(int)
    Xc["is_quiet_hours"] = Xc["is_quiet_hours"].astype(int)

    action_labels = df["action"].to_numpy()
    classes = np.unique(action_labels)

    propensity_prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["failure_reason"]),
            ("num", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
        ],
    )
    propensity_pipe = Pipeline(
        [("prep", propensity_prep), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]
    )

    oof_action_proba = cross_val_predict(
        propensity_pipe, Xc, action_labels, cv=cv, method="predict_proba"
    )
    class_index = {c: i for i, c in enumerate(classes)}
    propensities = np.array(
        [oof_action_proba[i, class_index[a]] for i, a in enumerate(action_labels)]
    )

    action_counts = pd.Series(action_labels).value_counts().to_dict()
    min_class_count = min(action_counts.values())

    # Clip small propensities before inverting to avoid exploding weights.
    clipped_propensities = np.clip(propensities, 0.02, None)
    weights = np.clip(1.0 / clipped_propensities, None, 20.0)
    ess = float((weights.sum() ** 2) / (weights**2).sum())

    propensity_mean_by_action = {
        a: float(propensities[action_labels == a].mean()) for a in classes
    }

    reliable = ess >= 0.5 * len(df) and min_class_count >= 20
    diagnostic = {
        "action_counts": {k: int(v) for k, v in action_counts.items()},
        "propensity_min": float(propensities.min()),
        "propensity_max": float(propensities.max()),
        "propensity_mean_by_action": propensity_mean_by_action,
        "effective_sample_size": ess,
        "n_rows": len(df),
        "considered_reliable": bool(reliable),
    }

    if not reliable:
        diagnostic["note"] = (
            "Propensity estimates are treated as UNRELIABLE: effective sample "
            f"size ({ess:.1f}/{len(df)}) and/or the smallest action count "
            f"({min_class_count}) are too small for stable inverse-propensity "
            "weights. Reporting the weighted-CV comparison anyway for "
            "transparency, but the primary model remains the unweighted one."
        )

    # Manual weighted-CV loop (same folds as the unweighted candidates) so
    # sample_weight can be passed into LogisticRegression.fit explicitly.
    oof_weighted = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
        pipe.fit(
            X.iloc[train_idx],
            y[train_idx],
            clf__sample_weight=weights[train_idx],
        )
        oof_weighted[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]

    weighted_auc = float(roc_auc_score(y, oof_weighted))
    diagnostic["logistic_regression_roc_auc_unweighted"] = float(unweighted_auc)
    diagnostic["logistic_regression_roc_auc_ips_weighted"] = weighted_auc
    diagnostic["ips_changed_auc_by"] = weighted_auc - float(unweighted_auc)

    return diagnostic


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df, X, y = load_training_data()

    print(f"Loaded {len(df)} logged transactions from {DATA_PATH}")
    print("Action counts:\n", df["action"].value_counts(), sep="")
    print(f"Overall recovery rate: {y.mean():.3f}\n")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }

    cv_results = []
    for name, estimator in candidates.items():
        metrics, oof_proba = evaluate_candidate(name, estimator, X, y, cv)
        cv_results.append(metrics)
        print(f"{name}: roc_auc={metrics['roc_auc']:.4f} log_loss={metrics['log_loss']:.4f} brier={metrics['brier']:.4f}")

    best = max(cv_results, key=lambda r: r["roc_auc"])
    best_name = best["model"]
    print(f"\nSelected model (highest CV ROC-AUC): {best_name}")

    final_pipe = Pipeline([("prep", build_preprocessor()), ("clf", candidates[best_name])])
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, MODEL_DIR / "recovery_model.joblib")

    print("\nRunning optional IPS diagnostic...")
    ips_diagnostic = run_ips_diagnostic(
        df, X, y, cv, unweighted_auc=next(r["roc_auc"] for r in cv_results if r["model"] == "logistic_regression")
    )
    print(json.dumps(ips_diagnostic, indent=2))

    metadata = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_training_rows": len(df),
        "action_counts": {k: int(v) for k, v in df["action"].value_counts().to_dict().items()},
        "overall_recovery_rate": float(y.mean()),
        "feature_columns": FEATURE_COLUMNS,
        "cv_results": cv_results,
        "selected_model": best_name,
        "ips_experiment": ips_diagnostic,
    }
    with open(MODEL_DIR / "train_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR / 'recovery_model.joblib'}")
    print(f"Saved metadata to {MODEL_DIR / 'train_metadata.json'}")


if __name__ == "__main__":
    main()
