"""
decision_insights.py

Pure, deterministic, presentation/analysis-only helpers for the Day 5
RecoverIQ Decision Assistant layer. Everything here reads already-computed
DecisionResult fields (or already-aggregated dashboard values) and returns
plain data. Nothing here calls DecisionEngine, PolicyEngine, or
ActionExecutor, and nothing here can alter a DecisionResult, final_action,
or any other decision-authority state -- these exist so Gemini narrates
numbers Python already computed, rather than being asked to compute them
itself.

Extracted out of app.py (rather than defined inline) specifically so they
can be unit tested the same way every other src/ module is tested, without
importing the Streamlit script itself.
"""

from typing import Optional


def compute_deterministic_insights(decision) -> dict:
    """
    decision: a DecisionResult (or any object exposing .ml_scores,
    .eligible_actions, .final_action, .expected_recovery_values).

    Two genuinely different concepts, kept distinct on purpose:

    - highest_scoring_action: ranked by raw ML score (decision.ml_scores)
      only, ignoring eligibility entirely -- "what did the model like
      best, whether or not policy would allow it."

    - strongest_eligible_alternative: ranked by expected_recovery_value
      (amount x probability -- the project's actual revenue objective,
      per Day 2's expected-recovery-value framing), among eligible actions
      OTHER than final_action. This can legitimately differ from "the
      eligible action with the next-highest ML score," because amount is
      constant per transaction but this field exists specifically to
      answer "what would the next-best REVENUE choice have been," not
      "what would the model have scored next-highest."
    """
    ml_scores = decision.ml_scores
    highest_scoring_action = max(ml_scores, key=ml_scores.get)
    eligible = set(decision.eligible_actions)

    alternatives = [a for a in eligible if a != decision.final_action]
    if alternatives:
        strongest_alt = max(alternatives, key=lambda a: decision.expected_recovery_values[a])
        strongest_alt_score = round(ml_scores[strongest_alt], 4)
        strongest_alt_value = round(decision.expected_recovery_values[strongest_alt], 2)
    else:
        strongest_alt = strongest_alt_score = strongest_alt_value = None

    return {
        "highest_scoring_action": highest_scoring_action,
        "highest_scoring_action_score": round(ml_scores[highest_scoring_action], 4),
        "highest_scoring_action_expected_recovery_value": round(
            decision.expected_recovery_values[highest_scoring_action], 2
        ),
        "was_highest_scoring_action_blocked_by_policy": highest_scoring_action not in eligible,
        "eligible_action_count": len(eligible),
        "strongest_eligible_alternative": strongest_alt,
        "strongest_eligible_alternative_score": strongest_alt_score,
        "strongest_eligible_alternative_expected_recovery_value": strongest_alt_value,
    }


def build_batch_summary(
    decisions,
    transactions_processed: int,
    revenue_at_risk: float,
    revenue_recovered: float,
    recovery_rate: float,
    passive_wait_baseline_pct_of_oracle_revenue: float,
    naive_historical_baseline_pct_of_oracle_revenue: float,
    full_comparison_summary: Optional[dict],
) -> dict:
    """
    Pure aggregation of already-computed numbers into one dict for the
    "Summarize this recovery batch" feature. No DecisionEngine/
    PolicyEngine call, no recomputation of oracle/system metrics -- every
    argument here is a value the caller (app.py) already computed.

    revenue_at_risk is NOT prefixed simulated_ -- it is the total
    transaction amount at stake, independent of any Bernoulli draw.
    revenue_recovered/recovery_rate ARE prefixed simulated_ -- they depend
    on the seeded demonstration outcome. oracle_revenue/system_revenue/
    pct_of_oracle_revenue/policy_override_rate come straight from
    full_comparison_summary (outputs/full_policy_comparison.json) and are
    never recomputed here.
    """
    intervention_counts: dict = {}
    for d in decisions:
        label = d.final_action if d.final_action is not None else "none (manual review)"
        intervention_counts[label] = intervention_counts.get(label, 0) + 1

    batch = {
        "summary_type": "batch",
        "transactions_processed": int(transactions_processed),
        "revenue_at_risk": round(float(revenue_at_risk), 2),
        "simulated_revenue_recovered": round(float(revenue_recovered), 2),
        "simulated_recovery_rate": round(float(recovery_rate), 4),
        "intervention_counts": intervention_counts,
        "passive_wait_baseline_pct_of_oracle_revenue": round(
            passive_wait_baseline_pct_of_oracle_revenue, 4
        ),
        "naive_historical_baseline_pct_of_oracle_revenue": round(
            naive_historical_baseline_pct_of_oracle_revenue, 4
        ),
    }
    if full_comparison_summary is not None:
        batch.update({
            "oracle_revenue": round(float(full_comparison_summary["oracle_revenue"]), 2),
            "system_revenue": round(float(full_comparison_summary["system_revenue"]), 2),
            "pct_of_oracle_revenue": full_comparison_summary["pct_of_oracle_revenue"],
            "policy_override_rate": full_comparison_summary["policy_override_rate"],
        })
    batch["note"] = (
        "simulated_* fields come from one seeded Bernoulli draw per "
        "transaction (a demo simulation, not observed real payments). "
        "revenue_at_risk is the total transaction amount, not simulated. "
        "oracle_revenue/system_revenue/pct_of_oracle_revenue/"
        "policy_override_rate come from a deterministic counterfactual "
        "evaluation against a synthetic holdout set with hidden "
        "true-probability labels, computed by run_full_policy_evaluation.py "
        "-- neither set is a production or real-world performance claim, "
        "and pct_of_oracle_revenue is a revenue-share metric, not a "
        "recovery rate."
    )
    return batch
