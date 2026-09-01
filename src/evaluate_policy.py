"""
evaluate_policy.py

Counterfactual policy evaluation on data/holdout_full_matrix.csv ONLY.
This is the only place the true_prob_* / best_action columns are ever read.

Compares six policies for each holdout transaction:
  - oracle          : best_action from the true_prob_* columns (upper bound)
  - random          : uniform random action
  - always_retry    : always picks "retry"
  - historical      : replays simulator.logging_policy.choose_logged_action
                       row-by-row with a seeded RNG (NOT the marginal
                       historical action distribution)
  - raw_model       : trained model's argmax over all 5 actions, unmasked
  - masked_model    : same model scores, but "wait" is only eligible when
                       active_promise_to_pay == True (deterministic rule
                       applied AFTER scoring, no retraining)

Regret for a chosen action = best_action_true_prob - true_prob_<chosen>

Run: python evaluate_policy.py   (from the src/ directory; requires
train_model.py to have been run first)
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from features import ACTIONS, add_action, to_model_matrix

sys.path.append(str(Path(__file__).parent.parent / "simulator"))
from logging_policy import choose_logged_action

SEED = 42
DATA_PATH = Path(__file__).parent.parent / "data" / "holdout_full_matrix.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "recovery_model.joblib"
OUT_DIR = Path(__file__).parent.parent / "outputs"


def score_all_actions(model, df):
    """Return a DataFrame [n_rows x len(ACTIONS)] of predicted P(recovered)."""
    scores = {}
    for action in ACTIONS:
        X = to_model_matrix(add_action(df, action))
        scores[action] = model.predict_proba(X)[:, 1]
    return pd.DataFrame(scores, index=df.index)


def historical_policy(df, seed=SEED):
    rng = np.random.default_rng(seed)
    actions = [choose_logged_action(row.to_dict(), rng) for _, row in df.iterrows()]
    return pd.Series(actions, index=df.index, name="action")


def random_policy(df, seed=SEED):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(ACTIONS, size=len(df)), index=df.index, name="action")


def always_retry_policy(df):
    return pd.Series(["retry"] * len(df), index=df.index, name="action")


def raw_model_policy(scores):
    return scores.idxmax(axis=1)


def masked_model_policy(scores, df):
    """Deterministic post-scoring rule: wait is only eligible when the
    customer has an active promise to pay. No retraining involved."""
    masked = scores.copy()
    ineligible = ~df["active_promise_to_pay"]
    masked.loc[ineligible, "wait"] = -np.inf
    return masked.idxmax(axis=1)


def true_prob_lookup(df, actions):
    """TRUE holdout recovery probability for the action chosen on each row."""
    cols = df[[f"true_prob_{a}" for a in ACTIONS]].to_numpy()
    action_to_col = {a: i for i, a in enumerate(ACTIONS)}
    col_idx = actions.map(action_to_col).to_numpy()
    return cols[np.arange(len(df)), col_idx]


def wait_selected_when_ptp_false(actions, df):
    ineligible = ~df["active_promise_to_pay"]
    if ineligible.sum() == 0:
        return 0.0
    return float((actions[ineligible] == "wait").mean())


def summarize(name, df, actions, oracle_total_revenue):
    chosen_true_prob = true_prob_lookup(df, actions)
    regret = df["best_action_true_prob"].to_numpy() - chosen_true_prob
    expected_revenue = chosen_true_prob * df["amount"].to_numpy()

    result = {
        "policy": name,
        "mean_true_recovery_prob": float(np.mean(chosen_true_prob)),
        "mean_regret": float(np.mean(regret)),
        "total_regret": float(np.sum(regret)),
        "pct_oracle_match": float(np.mean(actions.to_numpy() == df["best_action"].to_numpy())),
        "total_expected_revenue": float(np.sum(expected_revenue)),
        "pct_of_oracle_revenue": float(np.sum(expected_revenue) / oracle_total_revenue),
        "action_distribution": {k: int(v) for k, v in actions.value_counts().to_dict().items()},
    }
    if name in ("raw_model", "masked_model"):
        result["wait_selected_rate_when_ptp_false"] = wait_selected_when_ptp_false(actions, df)
    return result


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    print(f"Loaded {len(df)} holdout transactions from {DATA_PATH}")
    print("Oracle best_action distribution:\n", df["best_action"].value_counts(), sep="")

    scores = score_all_actions(model, df)

    oracle_actions = df["best_action"]
    oracle_total_revenue = float((df["best_action_true_prob"] * df["amount"]).sum())

    policies = {
        "oracle": oracle_actions,
        "random": random_policy(df),
        "always_retry": always_retry_policy(df),
        "historical": historical_policy(df),
        "raw_model": raw_model_policy(scores),
        "masked_model": masked_model_policy(scores, df),
    }

    results = [summarize(name, df, actions, oracle_total_revenue) for name, actions in policies.items()]
    results_df = pd.DataFrame(results)

    pd.set_option("display.width", 160)
    display_cols = [
        "policy", "mean_true_recovery_prob", "mean_regret", "pct_oracle_match",
        "pct_of_oracle_revenue",
    ]
    print("\n=== Policy comparison (holdout, ground-truth counterfactual) ===")
    print(results_df[display_cols].to_string(index=False))

    print("\n=== WAIT selected when active_promise_to_pay == False ===")
    for r in results:
        if "wait_selected_rate_when_ptp_false" in r:
            print(f"{r['policy']}: {r['wait_selected_rate_when_ptp_false']:.3f} "
                  f"({int(r['wait_selected_rate_when_ptp_false'] * (~df['active_promise_to_pay']).sum())} "
                  f"/ {(~df['active_promise_to_pay']).sum()} eligible rows)")

    print("\n=== Action distributions ===")
    for r in results:
        print(f"{r['policy']}: {r['action_distribution']}")

    results_df.to_csv(OUT_DIR / "policy_comparison.csv", index=False)
    with open(OUT_DIR / "policy_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results_df["policy"], results_df["mean_regret"], color="#4C72B0")
    ax.set_ylabel("Mean regret (oracle_prob - chosen_prob)")
    ax.set_title("RecoverIQ: policy comparison on holdout (lower is better)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "regret_by_policy.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
