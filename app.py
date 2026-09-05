"""
app.py

RecoverIQ Day 4 dashboard. Pure presentation/observability layer over the
existing Day 3 deterministic system -- it does not implement any ML
scoring, policy rule, or execution logic itself. It only:

  1. loads data/holdout_full_matrix.csv (read-only)
  2. calls DecisionEngine.decide() (via src/batch_runner.run_batch) and
     ActionExecutor.execute_decision() for every holdout transaction
  3. samples one reproducible simulated 0/1 outcome per transaction from
     the holdout's own true_prob_<final_action> column, so the dashboard
     can show concrete recovered/not-recovered counts instead of only
     probabilities
  4. displays the results

No LLM calls. No new eligibility rules. src/policy_engine.py's thresholds
and src/features.py are not touched or reimplemented here.

Run: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from action_executor import ActionExecutor
from batch_runner import run_batch
from decision_engine import DecisionEngine

HOLDOUT_PATH = Path(__file__).parent / "data" / "holdout_full_matrix.csv"
FULL_COMPARISON_JSON = Path(__file__).parent / "outputs" / "full_policy_comparison.json"

TRANSACTION_FIELDS = [
    "transaction_id", "amount", "failure_reason",
    "prior_successful_payments", "prior_failed_payments",
    "prior_recovery_attempts", "hours_since_failure",
    "active_promise_to_pay", "contact_count_last_7_days", "is_quiet_hours",
]

OUTCOME_SAMPLING_SEED = 42

# Presentation-only color maps -- purely cosmetic, do not affect any
# computed value. Fixed and consistent across reruns/sessions.
ACTION_COLORS = {
    "retry": "#4C72B0",
    "payment_link": "#55A868",
    "reminder": "#C44E52",
    "escalate": "#8172B2",
    "wait": "#CCB974",
    "none (manual review)": "#7F7F7F",
}

OUTCOME_COLORS = {
    "Recovered": "#2CA02C",
    "Not Recovered": "#D62728",
    "Held for Manual Review": "#9E9E9E",
    "Rejected at Execution": "#BDBDBD",
}

METRIC_ACCENT_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

st.set_page_config(page_title="RecoverIQ", layout="wide")


def row_to_transaction_dict(row: pd.Series) -> dict:
    d = {k: row[k] for k in TRANSACTION_FIELDS}
    d["active_promise_to_pay"] = bool(d["active_promise_to_pay"])
    d["is_quiet_hours"] = bool(d["is_quiet_hours"])
    for k in ["prior_successful_payments", "prior_failed_payments",
              "prior_recovery_attempts", "contact_count_last_7_days"]:
        d[k] = int(d[k])
    return d


@st.cache_data
def load_holdout() -> pd.DataFrame:
    return pd.read_csv(HOLDOUT_PATH)


@st.cache_resource
def get_engine() -> DecisionEngine:
    return DecisionEngine()


@st.cache_resource
def get_executor() -> ActionExecutor:
    return ActionExecutor()


@st.cache_data
def load_full_comparison_summary() -> dict:
    """Reads outputs/full_policy_comparison.json as-is. Returns None if the
    file doesn't exist -- the dashboard must not recompute these numbers
    itself, only display what run_full_policy_evaluation.py already produced."""
    if not FULL_COMPARISON_JSON.exists():
        return None
    with open(FULL_COMPARISON_JSON) as f:
        return json.load(f)


@st.cache_data
def compute_results():
    """
    Runs the full live decision path over every holdout transaction once,
    plus one reproducible simulated 0/1 outcome per transaction.

    Outcome sampling: a single np.random.default_rng(42) draw is taken for
    EVERY transaction, in holdout row order, regardless of whether that
    transaction ends up with a final_action -- this keeps the draw-to-row
    alignment stable even if a future policy change alters which
    transactions get decided vs. held/blocked. The draw is only consumed
    (compared against true_prob_<final_action>) when a real action was
    actually executed; otherwise it is discarded.

    This is a SIMULATED outcome, not an observed one -- there is no real
    payment event here, only a Bernoulli draw against the holdout's own
    hidden true_prob_<action> column.
    """
    holdout = load_holdout()
    engine = get_engine()
    executor = get_executor()

    transactions = [row_to_transaction_dict(row) for _, row in holdout.iterrows()]
    decisions = run_batch(transactions, engine)
    executions = [executor.execute_decision(d) for d in decisions]

    rng = np.random.default_rng(OUTCOME_SAMPLING_SEED)
    records = []
    for (_, row), decision, execution in zip(holdout.iterrows(), decisions, executions):
        draw = rng.random()  # always drawn, so row->draw alignment never shifts

        realized_outcome = None
        true_prob_for_chosen = None
        if decision.status == "decided" and decision.final_action is not None:
            true_prob_for_chosen = float(row[f"true_prob_{decision.final_action}"])
            if execution.status == "executed":
                realized_outcome = int(draw < true_prob_for_chosen)
                outcome_category = "Recovered" if realized_outcome == 1 else "Not Recovered"
            else:
                outcome_category = "Rejected at Execution"
        elif decision.status == "held_for_manual_review":
            outcome_category = "Held for Manual Review"
        else:
            outcome_category = f"Other ({decision.status})"

        records.append({
            "transaction_id": decision.transaction_id,
            "amount": float(row["amount"]),
            "final_action": decision.final_action,
            "decision_status": decision.status,
            "execution_status": execution.status,
            "realized_outcome": realized_outcome,
            "true_prob_for_chosen_action": true_prob_for_chosen,
            "outcome_category": outcome_category,
        })

    results_df = pd.DataFrame(records)
    return decisions, executions, results_df


holdout_df = load_holdout()
decisions, executions, results_df = compute_results()
decisions_by_id = {d.transaction_id: d for d in decisions}
executions_by_id = {e.transaction_id: e for e in executions}
holdout_by_id = holdout_df.set_index("transaction_id")
results_by_id = results_df.set_index("transaction_id")

summary = load_full_comparison_summary()

st.title("RecoverIQ — Decision & Policy Dashboard")
st.caption(
    "Presentation layer over the existing Day 3 deterministic system. "
    "All scoring, policy, and execution logic below is called from src/ — "
    "nothing is recomputed or reimplemented in this file."
)

# ---------------------------------------------------------------------
# 1. Top-level metrics
# ---------------------------------------------------------------------
st.header("1. Top-level metrics")

revenue_at_risk = results_df["amount"].sum()
revenue_recovered = results_df.loc[results_df["outcome_category"] == "Recovered", "amount"].sum()
recovery_rate = revenue_recovered / revenue_at_risk if revenue_at_risk else 0.0

st.markdown(f"""
<style>
.st-key-top_metrics_row div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"] {{
    border-left: 4px solid {METRIC_ACCENT_COLORS[0]};
    padding-left: 12px;
}}
.st-key-top_metrics_row div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] {{
    border-left: 4px solid {METRIC_ACCENT_COLORS[1]};
    padding-left: 12px;
}}
.st-key-top_metrics_row div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] {{
    border-left: 4px solid {METRIC_ACCENT_COLORS[2]};
    padding-left: 12px;
}}
.st-key-top_metrics_row div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] {{
    border-left: 4px solid {METRIC_ACCENT_COLORS[3]};
    padding-left: 12px;
}}
</style>
""", unsafe_allow_html=True)

with st.container(key="top_metrics_row"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue at Risk", f"₹{revenue_at_risk:,.0f}")
    c2.metric("Revenue Recovered (simulated)", f"₹{revenue_recovered:,.0f}")
    c3.metric("Recovery Rate", f"{recovery_rate:.1%}")
    c4.metric("Transactions Processed", f"{len(results_df)}")

if summary is not None:
    cc1, cc2 = st.columns(2)
    cc1.info(f"**{summary['pct_of_oracle_revenue']:.1%} of oracle revenue** "
              f"(from outputs/full_policy_comparison.json)")
    cc2.info(f"**{summary['policy_override_rate']:.1%} policy override rate** "
              f"(from outputs/full_policy_comparison.json)")
else:
    st.warning(
        "outputs/full_policy_comparison.json not found — run "
        "`python run_full_policy_evaluation.py` first to populate the "
        "oracle-comparison and override-rate callouts."
    )

st.caption(
    "Revenue Recovered / Recovery Rate are based on a SIMULATED outcome "
    "(a single seeded Bernoulli draw per transaction against the holdout's "
    "hidden true_prob_<final_action>), not an observed payment result."
)

# ---------------------------------------------------------------------
# 2. Interventions breakdown
# ---------------------------------------------------------------------
st.header("2. Interventions breakdown")
st.caption("final_action chosen across all processed transactions.")

action_labels = [d.final_action if d.final_action is not None else "none (manual review)" for d in decisions]
action_counts = pd.Series(action_labels).value_counts()  # value_counts() sorts descending by count already

assert action_counts.sum() == len(decisions), (
    f"Interventions breakdown counts ({action_counts.sum()}) do not sum to "
    f"the number of processed transactions ({len(decisions)})"
)
print(f"[sanity check] interventions breakdown sums to {action_counts.sum()} "
      f"(expected {len(decisions)})")

fig_actions, ax_actions = plt.subplots(figsize=(8, 4))
bar_colors = [ACTION_COLORS.get(action, "#7F7F7F") for action in action_counts.index]
bars = ax_actions.bar(action_counts.index, action_counts.values, color=bar_colors)
ax_actions.bar_label(bars, padding=3)
ax_actions.set_ylabel("Count")
ax_actions.set_xlabel("final_action")
plt.xticks(rotation=15)
st.pyplot(fig_actions)
plt.close(fig_actions)

# ---------------------------------------------------------------------
# 3. Recovery outcomes breakdown
# ---------------------------------------------------------------------
st.header("3. Recovery outcomes breakdown")
st.caption(
    "Derived from each transaction's decision_status and execution_status "
    "(plus the simulated outcome draw where an action was actually executed)."
)
outcome_counts = results_df["outcome_category"].value_counts()

fig_outcomes, ax_outcomes = plt.subplots(figsize=(8, 4))
outcome_bar_colors = [OUTCOME_COLORS.get(cat, "#9E9E9E") for cat in outcome_counts.index]
outcome_bars = ax_outcomes.bar(outcome_counts.index, outcome_counts.values, color=outcome_bar_colors)
ax_outcomes.bar_label(outcome_bars, padding=3)
ax_outcomes.set_ylabel("Count")
plt.xticks(rotation=15)
st.pyplot(fig_outcomes)
plt.close(fig_outcomes)

# ---------------------------------------------------------------------
# 4. Transaction trace view
# ---------------------------------------------------------------------
st.header("4. Transaction trace view")

selected_id = st.selectbox("transaction_id", results_df["transaction_id"].tolist())

decision = decisions_by_id[selected_id]
execution = executions_by_id[selected_id]
result_row = results_by_id.loc[selected_id]
holdout_row = holdout_by_id.loc[selected_id]

st.subheader("Transaction details")
tcol1, tcol2 = st.columns(2)
with tcol1:
    st.markdown("**From DecisionResult:**")
    st.write({"amount": decision.amount, "failure_reason": decision.failure_reason})
with tcol2:
    st.markdown("**From the original transaction record** "
                "(context fields not carried on DecisionResult itself):")
    st.write({
        "prior_successful_payments": int(holdout_row["prior_successful_payments"]),
        "prior_failed_payments": int(holdout_row["prior_failed_payments"]),
        "prior_recovery_attempts": int(holdout_row["prior_recovery_attempts"]),
        "hours_since_failure": float(holdout_row["hours_since_failure"]),
        "active_promise_to_pay": bool(holdout_row["active_promise_to_pay"]),
        "contact_count_last_7_days": int(holdout_row["contact_count_last_7_days"]),
        "is_quiet_hours": bool(holdout_row["is_quiet_hours"]),
    })

st.subheader("ML scores and expected recovery value (all 5 actions)")
scores_df = pd.DataFrame({
    "action": list(decision.ml_scores.keys()),
    "ml_score": list(decision.ml_scores.values()),
    "expected_recovery_value": [decision.expected_recovery_values[a] for a in decision.ml_scores],
})
st.dataframe(scores_df, width="stretch", hide_index=True)

if decision.policy_override:
    st.error(
        f"⚠️ **Policy overrode the model**: ML recommended **{decision.model_recommendation}**, "
        f"policy selected **{decision.final_action}** instead ({decision.override_reason})."
    )
else:
    st.success("✅ Policy agreed with the model's recommendation.")

st.subheader("Model recommendation vs final action")
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("model_recommendation", decision.model_recommendation or "—")
mcol2.metric("final_action", decision.final_action or "— (none)")
mcol3.metric("policy_override", str(decision.policy_override))

st.subheader("Policy checks (all 5 actions)")
policy_df = pd.DataFrame(decision.policy_checks)
policy_df["allowed"] = policy_df["allowed"].map({True: "✅ Allowed", False: "🚫 Blocked"})
st.dataframe(policy_df, width="stretch", hide_index=True)

st.subheader("Decision & execution status")
st.write({
    "decision_status": decision.status,
    "execution_status": execution.status,
    "flagged_for_review": decision.flagged_for_review,
    "flagged_for_review_reason": decision.flagged_for_review_reason,
})

st.subheader("Realized outcome (simulated)")
st.write({
    "outcome_category": result_row["outcome_category"],
    "realized_outcome": result_row["realized_outcome"],
    "true_prob_for_chosen_action": result_row["true_prob_for_chosen_action"],
})
