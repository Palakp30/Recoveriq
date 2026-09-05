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
  5. (Day 5, optional, on-demand) explains an already-finalized decision
     in plain English via src/llm_explainer.explain_decision() -- this
     never influences final_action; it only narrates a decision that
     DecisionEngine/PolicyEngine already made.

No new eligibility rules. src/policy_engine.py's thresholds and
src/features.py are not touched or reimplemented here. The LLM layer
never calls ActionExecutor and never constructs a PolicyCheckResult.

Run: streamlit run app.py
"""

import json
import os
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
from decision_insights import build_batch_summary, compute_deterministic_insights
from llm_explainer import explain_decision

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

METRIC_ACCENT_COLORS = ["#0B3D5C", "#0E7C7B", "#3B6E8F", "#6B7280"]  # navy, teal, slate-blue, grey

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


@st.cache_data(show_spinner="Asking RecoverIQ Decision Assistant...")
def get_cached_explanation(context: dict, _api_key: str) -> str:
    """
    Cached per distinct `context` (i.e. per transaction_id's recommendation
    context), so re-rendering the trace view or re-selecting the same
    transaction never re-calls the API. `_api_key` is underscore-prefixed
    so Streamlit excludes it from the cache key/hash (it's a credential,
    not part of the "data" identity of the request).
    """
    return explain_decision(context, _api_key)


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

with st.sidebar:
    st.markdown("### About RecoverIQ")
    st.caption(
        "RecoverIQ is a bounded, auditable payment-recovery decision system: "
        "an ML model scores five candidate recovery actions per failed "
        "transaction, a deterministic policy engine enforces eligibility "
        "rules, and only policy-approved actions ever execute. This "
        "dashboard is a read-only presentation layer over that Day 3 "
        "system — it never reimplements scoring or policy logic itself. "
        "An optional Gemini-powered layer can explain any already-finalized "
        "decision in plain English, but has no authority to change it."
    )
    st.divider()
    st.markdown("### Transaction lookup")
    selected_id = st.selectbox("transaction_id", results_df["transaction_id"].tolist())

st.markdown(
    """
    <div style="padding: 0.5rem 0 0.75rem 0;">
        <h1 style="margin-bottom: 0.2rem; color: #0B2545;">RecoverIQ</h1>
        <p style="font-size: 1.05rem; color: #4B5563; margin-top: 0;">
            Decision &amp; Policy Dashboard — a read-only presentation layer over the
            existing Day 3 deterministic system. All scoring, policy, and execution
            logic below is called from src/, nothing is recomputed or reimplemented here.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
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
    st.caption(
        "These two figures are a COUNTERFACTUAL EVALUATION result, not a "
        "recovery rate and not a simulation: they come from running the "
        "real DecisionEngine/PolicyEngine/ActionExecutor stack once against "
        "a synthetic holdout set whose hidden true-probability labels are "
        "known (run_full_policy_evaluation.py). \"% of oracle revenue\" is "
        "a revenue-share metric compared to the best-possible action per "
        "transaction, not a probability of recovery."
    )
else:
    st.warning(
        "outputs/full_policy_comparison.json not found — run "
        "`python run_full_policy_evaluation.py` first to populate the "
        "oracle-comparison and override-rate callouts."
    )

st.caption(
    "Revenue Recovered / Recovery Rate above are a SIMULATED BATCH OUTCOME "
    "(a single seeded Bernoulli draw per transaction against the holdout's "
    "hidden true_prob_<final_action>) -- a demonstration, not an observed "
    "real payment result, and a different concept from the counterfactual "
    "evaluation metrics below."
)

# ---------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------
st.header("Baseline comparison")
st.caption(
    "\"Passive / WAIT baseline\" and \"Naive historical rule\" are computed "
    "directly from the holdout DataFrame's true_prob_* columns below -- no "
    "DecisionEngine or PolicyEngine involved. RecoverIQ's own numbers are "
    "read as-is from outputs/full_policy_comparison.json, not recomputed here."
)

total_amount = float(holdout_df["amount"].sum())
if summary is not None:
    oracle_revenue = float(summary["oracle_revenue"])
else:
    oracle_revenue = float((holdout_df["amount"] * holdout_df["best_action_true_prob"]).sum())

# "Passive / WAIT baseline": WAIT applied to every single transaction,
# regardless of active_promise_to_pay. This is the true_prob_wait column,
# amount-weighted. WAIT is a real candidate action (not literally "no
# action" from the system's perspective), so it's labeled precisely.
no_action_revenue = float((holdout_df["amount"] * holdout_df["true_prob_wait"]).sum())
no_action_recovery_rate = no_action_revenue / total_amount
no_action_pct_oracle = no_action_revenue / oracle_revenue


def deterministic_historical_action(row) -> str:
    """
    A DETERMINISTIC replay of simulator/logging_policy.py's actual branches
    (read from that file, not modified or imported) -- each stochastic
    rng.random() branch is collapsed to its majority/mode outcome instead of
    calling the original stochastic function:
      - active_promise_to_pay=True -> "wait"     (70% branch there, the majority)
      - prior_recovery_attempts==0 -> "retry"     (no randomness in the original)
      - prior_recovery_attempts==1 -> "reminder"  (no randomness in the original)
      - prior_recovery_attempts==2 -> "reminder"  (70% there, beats 30% payment_link)
      - prior_recovery_attempts>=3 -> "reminder"  (exact 50/50 vs escalate there;
        tie broken toward "reminder" since that's the empirically dominant
        action in the real logged data -- 397/800 rows in
        data/logged_transactions.csv)
    """
    if bool(row["active_promise_to_pay"]):
        return "wait"
    attempts = int(row["prior_recovery_attempts"])
    if attempts == 0:
        return "retry"
    else:
        return "reminder"  # covers attempts 1, 2 (mode), and >=3 (tie-break)


naive_actions = holdout_df.apply(deterministic_historical_action, axis=1)
naive_true_probs = pd.Series(
    [holdout_df.iloc[i][f"true_prob_{a}"] for i, a in enumerate(naive_actions)],
    index=holdout_df.index,
)
naive_revenue = float((holdout_df["amount"] * naive_true_probs).sum())
naive_recovery_rate = naive_revenue / total_amount
naive_pct_oracle = naive_revenue / oracle_revenue

if summary is not None:
    recoveriq_recovery_rate = summary["mean_chosen_true_prob"]
    recoveriq_pct_oracle = summary["pct_of_oracle_revenue"]
    recoveriq_revenue = summary["system_revenue"]
else:
    recoveriq_recovery_rate = recoveriq_pct_oracle = recoveriq_revenue = None

st.markdown(f"""
<style>
.st-key-recoveriq_standout {{
    background-color: #EAF6F5;
    border: 2px solid {METRIC_ACCENT_COLORS[1]};
    border-radius: 8px;
    padding: 0.75rem 1rem;
}}
</style>
""", unsafe_allow_html=True)

bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    st.markdown("**Passive / WAIT baseline**")
    st.caption("WAIT applied to every transaction")
    st.metric("Recovery rate", f"{no_action_recovery_rate:.1%}")
    st.metric("% of oracle revenue", f"{no_action_pct_oracle:.1%}")
    st.caption(f"₹{no_action_revenue:,.0f} revenue")
with bcol2:
    st.markdown("**Naive historical rule**")
    st.caption("deterministic replay of logging_policy.py")
    st.metric("Recovery rate", f"{naive_recovery_rate:.1%}")
    st.metric("% of oracle revenue", f"{naive_pct_oracle:.1%}")
    st.caption(f"₹{naive_revenue:,.0f} revenue")
with bcol3:
    with st.container(key="recoveriq_standout"):
        st.markdown("**RecoverIQ (full system)**")
        st.caption("DecisionEngine + PolicyEngine, all 5 rules")
        if recoveriq_pct_oracle is not None:
            st.metric("Recovery rate", f"{recoveriq_recovery_rate:.1%}")
            st.metric("% of oracle revenue", f"{recoveriq_pct_oracle:.1%}")
            st.caption(f"₹{recoveriq_revenue:,.0f} revenue")
        else:
            st.warning("Run run_full_policy_evaluation.py first.")


st.subheader("RecoverIQ Decision Assistant — Batch Summary")
st.caption(
    "Summarizes the metrics already shown above -- Gemini narrates these "
    "numbers, it does not calculate them."
)
if st.button("Summarize this recovery batch"):
    api_key_for_batch = os.getenv("GEMINI_API_KEY")
    if not api_key_for_batch:
        st.warning("GEMINI_API_KEY is not set (check .env) -- cannot summarize.")
    else:
        batch_context = build_batch_summary(
            decisions=decisions,
            transactions_processed=len(results_df),
            revenue_at_risk=revenue_at_risk,
            revenue_recovered=revenue_recovered,
            recovery_rate=recovery_rate,
            passive_wait_baseline_pct_of_oracle_revenue=no_action_pct_oracle,
            naive_historical_baseline_pct_of_oracle_revenue=naive_pct_oracle,
            full_comparison_summary=summary,
        )
        # Primary UI is the product-level narrative; the raw JSON sent to
        # the assistant is secondary, collapsed by default.
        batch_summary_text = get_cached_explanation(batch_context, api_key_for_batch)
        st.info(batch_summary_text)
        st.caption("Explanation only — cannot change or execute any decision.")
        with st.expander("Batch metrics sent to the assistant (nothing is calculated by Gemini)"):
            st.json(batch_context)

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

# Horizontal layout reads more easily with count labels beside each bar.
# Reversed so the largest count (already first, via value_counts()'s
# descending sort) renders at the TOP of the chart.
labels_top_down = action_counts.index[::-1]
values_top_down = action_counts.values[::-1]
bar_colors = [ACTION_COLORS.get(action, "#7F7F7F") for action in labels_top_down]

fig_actions, ax_actions = plt.subplots(figsize=(8, 4))
bars = ax_actions.barh(labels_top_down, values_top_down, color=bar_colors)
ax_actions.bar_label(bars, padding=3)
ax_actions.set_xlabel("Count")
ax_actions.set_ylabel("final_action")
plt.tight_layout()
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

# Report which of the conceptually-possible categories had zero occurrences
# in this batch, rather than silently omitting them or fabricating a bar.
ALL_POSSIBLE_OUTCOME_CATEGORIES = [
    "Recovered", "Not Recovered", "Held for Manual Review", "Rejected at Execution",
]
zero_categories = [c for c in ALL_POSSIBLE_OUTCOME_CATEGORIES if c not in outcome_counts.index]
if zero_categories:
    st.caption(
        f"No transactions fell into: {', '.join(zero_categories)} "
        f"(0 occurrences in this batch)."
    )

labels_top_down = outcome_counts.index[::-1]
values_top_down = outcome_counts.values[::-1]
outcome_bar_colors = [OUTCOME_COLORS.get(cat, "#9E9E9E") for cat in labels_top_down]

fig_outcomes, ax_outcomes = plt.subplots(figsize=(8, 4))
outcome_bars = ax_outcomes.barh(labels_top_down, values_top_down, color=outcome_bar_colors)
ax_outcomes.bar_label(outcome_bars, padding=3)
ax_outcomes.set_xlabel("Count")
plt.tight_layout()
st.pyplot(fig_outcomes)
plt.close(fig_outcomes)

# ---------------------------------------------------------------------
# 4. Transaction trace view
# ---------------------------------------------------------------------
st.header("4. Transaction trace view")
st.caption(f"Showing **{selected_id}** — pick a different transaction_id in the sidebar.")

decision = decisions_by_id[selected_id]
execution = executions_by_id[selected_id]
result_row = results_by_id.loc[selected_id]
holdout_row = holdout_by_id.loc[selected_id]

st.subheader("Transaction details")

# Only fields that actually exist on DecisionResult (amount, failure_reason)
# or the original holdout transaction record (everything else) -- nothing
# invented. amount/failure_reason come from DecisionResult specifically;
# the rest aren't carried on DecisionResult, so they're read from the
# original transaction record instead.
txn_fields = {
    "Transaction ID": selected_id,
    "Amount": f"₹{decision.amount:,.2f}",
    "Failure reason": decision.failure_reason,
    "Prior successful payments": int(holdout_row["prior_successful_payments"]),
    "Prior failed payments": int(holdout_row["prior_failed_payments"]),
    "Prior recovery attempts": int(holdout_row["prior_recovery_attempts"]),
    "Hours since failure": float(holdout_row["hours_since_failure"]),
    "Active promise to pay": bool(holdout_row["active_promise_to_pay"]),
    "Contacts in last 7 days": int(holdout_row["contact_count_last_7_days"]),
    "Quiet hours": bool(holdout_row["is_quiet_hours"]),
}
txn_table = pd.DataFrame(
    {"Field": list(txn_fields.keys()), "Value": [str(v) for v in txn_fields.values()]}
)
st.dataframe(txn_table, hide_index=True, width="stretch")

with st.expander("Raw transaction record"):
    st.write({"amount": decision.amount, "failure_reason": decision.failure_reason})
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
    "ml_score": [round(v, 4) for v in decision.ml_scores.values()],
    "expected_recovery_value": [
        round(decision.expected_recovery_values[a], 2) for a in decision.ml_scores
    ],
})
st.dataframe(scores_df, width="stretch", hide_index=True)

st.subheader("Model → Policy → Final action")
if decision.policy_override:
    st.error(
        f"⚠️ **Policy overrode the model**: ML recommended **{decision.model_recommendation}**, "
        f"policy selected **{decision.final_action}** instead ({decision.override_reason})."
    )
else:
    st.success("✅ Policy agreed with the model's recommendation — no override.")

flow_col1, flow_arrow, flow_col2 = st.columns([2, 1, 2])
flow_col1.metric("Model recommended", decision.model_recommendation or "—")
flow_arrow.markdown(
    "<div style='text-align:center; font-size:1.8rem; padding-top:0.6rem;'>→</div>",
    unsafe_allow_html=True,
)
flow_col2.metric("Final action", decision.final_action or "— (none)")

st.metric("Policy override", "YES" if decision.policy_override else "NO")
if decision.policy_override:
    st.caption(f"Reason: {decision.override_reason}")

st.subheader("Policy checks (all 5 actions)")
policy_df = pd.DataFrame(decision.policy_checks)
policy_df["allowed"] = policy_df["allowed"].map({True: "✅ Allowed", False: "🚫 Blocked"})
st.dataframe(policy_df, width="stretch", hide_index=True)

st.subheader("RecoverIQ Decision Assistant")
st.caption(
    "AI-generated interpretation of the already-finalized decision above "
    "(powered by Gemini). Reads DecisionEngine.build_recommendation_context() "
    "plus the already-computed model_recommendation/final_action/"
    "policy_override -- never calls ActionExecutor, never constructs a "
    "PolicyCheckResult, and cannot change final_action."
)


def _build_rec_context() -> dict:
    # holdout_row came from holdout_by_id (indexed by transaction_id), so
    # "transaction_id" is its Series name, not a field -- add it back
    # before handing this to row_to_transaction_dict, which expects it
    # as one of TRANSACTION_FIELDS.
    holdout_row_dict = holdout_row.to_dict()
    holdout_row_dict["transaction_id"] = holdout_row.name
    txn_dict = row_to_transaction_dict(holdout_row_dict)
    ctx = get_engine().build_recommendation_context(txn_dict)
    # build_recommendation_context() itself never exposes these (it's a
    # pre-decision view, deliberately not touched here) -- merge them in
    # from the already-computed `decision` so the LLM can accurately
    # describe override status instead of inferring it from eligibility
    # alone. Deterministic derived insights are computed in Python (never
    # by Gemini) and merged in the same way.
    return {
        **ctx,
        "model_recommendation": decision.model_recommendation,
        "final_action": decision.final_action,
        "policy_override": decision.policy_override,
        "override_reason": decision.override_reason,
        **compute_deterministic_insights(decision),
    }


api_key = os.getenv("GEMINI_API_KEY")

if st.button("Explain this decision", key=f"explain_{selected_id}"):
    if not api_key:
        st.warning("GEMINI_API_KEY is not set (check .env) -- cannot generate an explanation.")
    else:
        rec_context = _build_rec_context()
        explanation = get_cached_explanation(rec_context, api_key)
        st.info(explanation)
        st.caption("Explanation only — cannot change or execute the decision.")

# Bounded question set -- NOT a free-text chatbot. Only these specific,
# pre-approved questions about the already-finalized decision are
# selectable; there is no way for a user to type an arbitrary instruction
# (e.g. "change the action to payment_link") through this UI.
BOUNDED_QUESTIONS = [
    "Why was this action selected?",
    "Why wasn't WAIT selected?",
    "Why was the model recommendation overridden?",
    "Which policy rule affected this decision?",
    "What was the strongest eligible alternative?",
]

with st.expander("Ask about this decision"):
    st.caption(
        "Bounded to the questions below -- RecoverIQ Decision Assistant "
        "cannot select, change, or execute an action."
    )
    selected_question = st.selectbox(
        "Choose a question", BOUNDED_QUESTIONS, key=f"question_{selected_id}"
    )
    if st.button("Ask", key=f"ask_{selected_id}"):
        if not api_key:
            st.warning("GEMINI_API_KEY is not set (check .env) -- cannot answer.")
        else:
            rec_context = _build_rec_context()
            rec_context["user_question"] = selected_question
            answer = get_cached_explanation(rec_context, api_key)
            st.info(answer)
            st.caption("Explanation only — cannot change or execute the decision.")

st.subheader("Decision & execution status")
scol1, scol2 = st.columns(2)
scol1.metric("decision_status", decision.status)
scol2.metric("execution_status", execution.status)
if decision.flagged_for_review:
    st.info(f"🚩 Flagged for review: {decision.flagged_for_review_reason}")

st.subheader("Simulated realized outcome")
st.caption(
    "One seeded Bernoulli draw (fixed seed) against the holdout's own "
    "true_prob_<final_action> column -- NOT an observed real payment result."
)
outcome_category_value = str(result_row["outcome_category"])
true_prob_value = result_row["true_prob_for_chosen_action"]

ocol1, ocol2 = st.columns(2)
ocol1.metric("Simulated outcome", outcome_category_value)
if pd.notna(true_prob_value):
    ocol2.metric("True probability of chosen action", f"{float(true_prob_value):.1%}")
else:
    ocol2.metric("True probability of chosen action", "—")
