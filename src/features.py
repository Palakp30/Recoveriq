"""
features.py

Shared feature engineering for RecoverIQ Day 2.

The SAME functions are used by train_model.py (fit) and evaluate_policy.py
(transform-only) so there is no train/serve feature mismatch. This file
contains no hidden reward logic and never references holdout
true_prob_* / best_action columns.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ACTIONS = ["retry", "payment_link", "reminder", "escalate", "wait"]

NUMERIC_FEATURES = [
    "amount",
    "prior_successful_payments",
    "prior_failed_payments",
    "prior_recovery_attempts",
    "hours_since_failure",
    "contact_count_last_7_days",
]

BOOLEAN_FEATURES = ["active_promise_to_pay", "is_quiet_hours"]

# "action" is included as a feature so one model can score every candidate
# action for a transaction (see evaluate_policy.score_all_actions).
CATEGORICAL_FEATURES = ["failure_reason", "action"]

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    One-hot encodes failure_reason/action; standardizes numeric/boolean
    columns (needed for LogisticRegression's solver to converge cleanly;
    harmless no-op for the scale-invariant GradientBoostingClassifier).
    Must be fit only on logged_transactions.csv (inside the training
    Pipeline) so no holdout information ever influences it.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
        ],
    )


def add_action(df, action_value):
    """Return a copy of df with a constant 'action' column set to action_value."""
    out = df.copy()
    out["action"] = action_value
    return out


def to_model_matrix(df):
    """Select feature columns in a fixed order; cast booleans to int."""
    out = df[FEATURE_COLUMNS].copy()
    out["active_promise_to_pay"] = out["active_promise_to_pay"].astype(int)
    out["is_quiet_hours"] = out["is_quiet_hours"].astype(int)
    return out
