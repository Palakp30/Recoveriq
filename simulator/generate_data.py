"""
generate_data.py

Produces two datasets:

1. logged_transactions.csv  (TRAINING DATA)
   - Generated using the biased logging_policy
   - Each row = one transaction, one action actually taken, ONE realized
     outcome (recovered / not recovered)
   - This is all the ML model in Day 2 is allowed to see.

2. holdout_full_matrix.csv  (EVALUATION DATA — ML model never trains on this)
   - A separate, non-overlapping set of transactions
   - For EACH transaction, we compute the TRUE recovery probability for
     EVERY action using the hidden reward function
   - This lets us evaluate: "if the model/policy picked action X, how good
     was that choice truly, vs the best possible action?"
   - This is what makes your counterfactual/off-policy evaluation honest.

Run: python generate_data.py
"""

import numpy as np
import pandas as pd
from reward_function import true_recovery_probability, full_counterfactual_matrix, ACTIONS
from logging_policy import choose_logged_action

SEED = 42
N_LOGGED = 800
N_HOLDOUT = 300

FAILURE_REASONS = ["insufficient_funds", "card_declined", "expired_card",
                    "network_error", "bank_timeout"]
FAILURE_WEIGHTS = [0.35, 0.25, 0.15, 0.15, 0.10]


def sample_transaction(txn_id: str, rng: np.random.Generator) -> dict:
    prior_success = int(rng.poisson(5))
    prior_fail = int(rng.poisson(1.2))
    prior_recovery_attempts = int(min(rng.poisson(1.0), 4))
    amount = float(np.round(np.exp(rng.normal(8.5, 1.0)), 2))  # log-normal, INR-ish scale
    failure_reason = rng.choice(FAILURE_REASONS, p=FAILURE_WEIGHTS)
    hours_since_failure = float(rng.exponential(30))
    active_ptp = bool(rng.random() < 0.12)
    contact_count_7d = int(rng.poisson(0.8))
    is_quiet_hours = bool(rng.random() < 0.2)

    return {
        "transaction_id": txn_id,
        "amount": amount,
        "failure_reason": failure_reason,
        "prior_successful_payments": prior_success,
        "prior_failed_payments": prior_fail,
        "prior_recovery_attempts": prior_recovery_attempts,
        "hours_since_failure": round(hours_since_failure, 1),
        "active_promise_to_pay": active_ptp,
        "contact_count_last_7_days": contact_count_7d,
        "is_quiet_hours": is_quiet_hours,
    }


def generate_logged_dataset(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n):
        txn = sample_transaction(f"LOG-{i:05d}", rng)
        action = choose_logged_action(txn, rng)
        true_p = true_recovery_probability(txn, action, rng)
        outcome = int(rng.random() < true_p)  # realized, stochastic outcome
        row = dict(txn)
        row["action_taken"] = action
        row["recovered"] = outcome
        rows.append(row)
    return pd.DataFrame(rows)


def generate_holdout_dataset(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n):
        txn = sample_transaction(f"HOLD-{i:05d}", rng)
        cf = full_counterfactual_matrix(txn, rng)
        row = dict(txn)
        for a in ACTIONS:
            row[f"true_prob_{a}"] = round(cf[a], 4)
        row["best_action"] = max(cf, key=cf.get)
        row["best_action_true_prob"] = round(max(cf.values()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    logged_df = generate_logged_dataset(N_LOGGED, rng)
    holdout_df = generate_holdout_dataset(N_HOLDOUT, rng)
    
    logged_df.to_csv("../data/logged_transactions.csv", index=False)
    holdout_df.to_csv("../data/holdout_full_matrix.csv", index=False)

    print("=== Logged (training) dataset ===")
    print(f"Rows: {len(logged_df)}")
    print("Action distribution (selection bias check):")
    print(logged_df["action_taken"].value_counts())
    print(f"Overall recovery rate: {logged_df['recovered'].mean():.3f}")
    print()
    print("=== Holdout (evaluation) dataset ===")
    print(f"Rows: {len(holdout_df)}")
    print("Best-action distribution (what an oracle would pick):")
    print(holdout_df["best_action"].value_counts())
    print(f"Mean best-action true recovery prob: {holdout_df['best_action_true_prob'].mean():.3f}")
