"""
run_full_policy_evaluation.py

Runs the ACTUAL live decision path (DecisionEngine -> PolicyEngine ->
ActionExecutor) against the full holdout set, end-to-end. Replaces
evaluate_policy.py's WAIT-only masked comparison with the true number
the full 5-rule system produces.

Read-only with respect to data/, models/, src/ -- only writes to outputs/.
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))

from decision_engine import DecisionEngine
from action_executor import ActionExecutor

HOLDOUT_PATH = Path(__file__).parent / "data" / "holdout_full_matrix.csv"
OUTPUT_CSV = Path(__file__).parent / "outputs" / "full_policy_comparison.csv"
OUTPUT_JSON = Path(__file__).parent / "outputs" / "full_policy_comparison.json"

TRANSACTION_FIELDS = [
    "transaction_id", "amount", "failure_reason",
    "prior_successful_payments", "prior_failed_payments",
    "prior_recovery_attempts", "hours_since_failure",
    "active_promise_to_pay", "contact_count_last_7_days", "is_quiet_hours",
]


def row_to_transaction_dict(row: pd.Series) -> dict:
    d = {k: row[k] for k in TRANSACTION_FIELDS}
    d["active_promise_to_pay"] = bool(d["active_promise_to_pay"])
    d["is_quiet_hours"] = bool(d["is_quiet_hours"])
    for k in ["prior_successful_payments", "prior_failed_payments",
              "prior_recovery_attempts", "contact_count_last_7_days"]:
        d[k] = int(d[k])
    return d


def main():
    holdout = pd.read_csv(HOLDOUT_PATH)
    engine = DecisionEngine()
    executor = ActionExecutor()

    rows = []
    for _, row in holdout.iterrows():
        txn_dict = row_to_transaction_dict(row)
        decision = engine.decide(txn_dict)
        exec_result = executor.execute_decision(decision)

        if decision.status != "decided" or decision.final_action is None:
            final_action = None
            chosen_true_prob = 0.0
        else:
            final_action = decision.final_action
            chosen_true_prob = float(row[f"true_prob_{final_action}"])

        rows.append({
            "transaction_id": row["transaction_id"],
            "amount": row["amount"],
            "best_action": row["best_action"],
            "best_true_prob": row["best_action_true_prob"],
            "model_recommendation": decision.model_recommendation,
            "final_action": final_action,
            "policy_override": decision.policy_override,
            "decision_status": decision.status,
            "execution_status": exec_result.status,
            "chosen_true_prob": chosen_true_prob,
        })

    res = pd.DataFrame(rows)
    res["regret"] = res["best_true_prob"] - res["chosen_true_prob"]
    res["oracle_match"] = res["final_action"] == res["best_action"]

    total_amount = res["amount"].sum()
    system_revenue = (res["amount"] * res["chosen_true_prob"]).sum()
    oracle_revenue = (res["amount"] * res["best_true_prob"]).sum()

    summary = {
        "n": len(res),
        "match_rate": round(res["oracle_match"].mean(), 4),
        "mean_regret": round(res["regret"].mean(), 4),
        "mean_chosen_true_prob": round(res["chosen_true_prob"].mean(), 4),
        "pct_of_oracle_revenue": round(system_revenue / oracle_revenue, 4),
        "system_revenue": round(system_revenue, 2),
        "oracle_revenue": round(oracle_revenue, 2),
        "total_amount_at_risk": round(total_amount, 2),
        "decision_status_breakdown": res["decision_status"].value_counts().to_dict(),
        "execution_status_breakdown": res["execution_status"].value_counts().to_dict(),
        "policy_override_rate": round(res["policy_override"].mean(), 4),
    }

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    res.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== FULL Day 3 stack vs oracle ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()