import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from decision_insights import build_batch_summary, compute_deterministic_insights
from schemas import DecisionResult

REPO_ROOT = Path(__file__).parent.parent
FULL_COMPARISON_JSON = REPO_ROOT / "outputs" / "full_policy_comparison.json"
HOLDOUT_PATH = REPO_ROOT / "data" / "holdout_full_matrix.csv"
MODEL_PATH = REPO_ROOT / "models" / "recovery_model.joblib"


def make_decision(**overrides):
    defaults = dict(
        transaction_id="TEST-INSIGHTS-1",
        timestamp="2026-01-01T00:00:00+00:00",
        status="decided",
        amount=1000.0,
        failure_reason="card_declined",
        ml_scores={
            "retry": 0.3, "payment_link": 0.5, "reminder": 0.2,
            "escalate": 0.4, "wait": 0.9,
        },
        expected_recovery_values={
            "retry": 300.0, "payment_link": 500.0, "reminder": 200.0,
            "escalate": 400.0, "wait": 900.0,
        },
        eligible_actions=["retry", "payment_link", "reminder", "escalate"],  # wait blocked
        rejected_actions=[{"action": "wait", "allowed": False,
                            "reason": "WAIT requires an active promise-to-pay."}],
        model_recommendation="wait",
        final_action="payment_link",
        policy_override=True,
        override_reason="WAIT requires an active promise-to-pay.",
    )
    defaults.update(overrides)
    return DecisionResult(**defaults)


class TestHighestScoringAction(unittest.TestCase):
    def test_highest_scoring_action_eligible(self):
        decision = make_decision(
            ml_scores={"retry": 0.3, "payment_link": 0.9, "reminder": 0.2,
                       "escalate": 0.4, "wait": 0.1},
            eligible_actions=["retry", "payment_link", "reminder", "escalate", "wait"],
            final_action="payment_link",
        )
        insights = compute_deterministic_insights(decision)
        self.assertEqual(insights["highest_scoring_action"], "payment_link")
        self.assertFalse(insights["was_highest_scoring_action_blocked_by_policy"])

    def test_highest_scoring_action_blocked(self):
        decision = make_decision(
            ml_scores={"retry": 0.3, "payment_link": 0.5, "reminder": 0.2,
                       "escalate": 0.4, "wait": 0.9},
            eligible_actions=["retry", "payment_link", "reminder", "escalate"],  # wait excluded
            final_action="payment_link",
        )
        insights = compute_deterministic_insights(decision)
        self.assertEqual(insights["highest_scoring_action"], "wait")
        self.assertTrue(insights["was_highest_scoring_action_blocked_by_policy"])


class TestStrongestEligibleAlternative(unittest.TestCase):
    def test_multiple_eligible_alternatives_ranked_by_expected_value_not_score(self):
        # payment_link has the higher ML score (0.6) but escalate has the
        # higher expected_recovery_value (600 vs 480) -- the alternative
        # must be picked by expected value (the revenue objective), not by
        # ML score.
        decision = make_decision(
            ml_scores={"retry": 0.3, "payment_link": 0.6, "reminder": 0.2,
                       "escalate": 0.5, "wait": 0.1},
            expected_recovery_values={"retry": 300.0, "payment_link": 480.0,
                                       "reminder": 200.0, "escalate": 600.0, "wait": 100.0},
            eligible_actions=["retry", "payment_link", "reminder", "escalate"],
            final_action="retry",
        )
        insights = compute_deterministic_insights(decision)
        self.assertEqual(insights["strongest_eligible_alternative"], "escalate")
        self.assertEqual(insights["strongest_eligible_alternative_expected_recovery_value"], 600.0)
        self.assertEqual(insights["strongest_eligible_alternative_score"], 0.5)
        self.assertNotEqual(insights["strongest_eligible_alternative"], "payment_link")

    def test_ml_score_ranking_differs_from_expected_value_ranking(self):
        decision = make_decision(
            ml_scores={"retry": 0.9, "payment_link": 0.6, "reminder": 0.2,
                       "escalate": 0.5, "wait": 0.1},
            expected_recovery_values={"retry": 100.0, "payment_link": 480.0,
                                       "reminder": 200.0, "escalate": 600.0, "wait": 50.0},
            eligible_actions=["payment_link", "reminder", "escalate"],
            final_action="payment_link",
        )
        insights = compute_deterministic_insights(decision)
        # Highest RAW ML score overall is "retry" (0.9) -- ignores eligibility.
        self.assertEqual(insights["highest_scoring_action"], "retry")
        self.assertTrue(insights["was_highest_scoring_action_blocked_by_policy"])
        # Strongest ELIGIBLE alternative (excluding final_action) ranked by
        # expected value is "escalate" (600) -- the two concepts disagree
        # on which action is "best," proving they are genuinely distinct.
        self.assertEqual(insights["strongest_eligible_alternative"], "escalate")
        self.assertEqual(insights["strongest_eligible_alternative_expected_recovery_value"], 600.0)
        self.assertNotEqual(
            insights["highest_scoring_action"], insights["strongest_eligible_alternative"]
        )

    def test_no_eligible_alternative(self):
        decision = make_decision(
            eligible_actions=["retry"],  # only the final_action itself is eligible
            final_action="retry",
        )
        insights = compute_deterministic_insights(decision)
        self.assertIsNone(insights["strongest_eligible_alternative"])
        self.assertIsNone(insights["strongest_eligible_alternative_score"])
        self.assertIsNone(insights["strongest_eligible_alternative_expected_recovery_value"])

    def test_final_action_excluded_from_alternatives(self):
        decision = make_decision(
            ml_scores={"retry": 0.9, "payment_link": 0.5, "reminder": 0.2,
                       "escalate": 0.4, "wait": 0.1},
            expected_recovery_values={"retry": 900.0, "payment_link": 500.0,
                                       "reminder": 200.0, "escalate": 400.0, "wait": 100.0},
            eligible_actions=["retry", "payment_link", "reminder", "escalate"],
            final_action="retry",  # retry wins on both score and value, but IS final_action
        )
        insights = compute_deterministic_insights(decision)
        self.assertNotEqual(insights["strongest_eligible_alternative"], "retry")
        self.assertEqual(insights["strongest_eligible_alternative"], "payment_link")


class TestHelperCannotAlterDecisionResult(unittest.TestCase):
    def test_decision_result_unchanged_after_call(self):
        decision = make_decision()
        ml_scores_before = dict(decision.ml_scores)
        expected_values_before = dict(decision.expected_recovery_values)
        eligible_before = list(decision.eligible_actions)
        final_action_before = decision.final_action
        policy_override_before = decision.policy_override

        compute_deterministic_insights(decision)

        self.assertEqual(decision.ml_scores, ml_scores_before)
        self.assertEqual(decision.expected_recovery_values, expected_values_before)
        self.assertEqual(decision.eligible_actions, eligible_before)
        self.assertEqual(decision.final_action, final_action_before)
        self.assertEqual(decision.policy_override, policy_override_before)


class TestBuildBatchSummary(unittest.TestCase):
    def test_field_naming_and_values(self):
        fake_decisions = [
            make_decision(final_action="retry"),
            make_decision(final_action="payment_link"),
            make_decision(final_action="retry"),
        ]
        comparison_summary = {
            "oracle_revenue": 1498542.16,
            "system_revenue": 1273814.56,
            "pct_of_oracle_revenue": 0.85,
            "policy_override_rate": 0.6567,
        }
        batch = build_batch_summary(
            decisions=fake_decisions,
            transactions_processed=3,
            revenue_at_risk=3000.0,
            revenue_recovered=1500.0,
            recovery_rate=0.5,
            passive_wait_baseline_pct_of_oracle_revenue=0.439,
            naive_historical_baseline_pct_of_oracle_revenue=0.590,
            full_comparison_summary=comparison_summary,
        )
        self.assertIn("revenue_at_risk", batch)
        self.assertNotIn("simulated_revenue_at_risk", batch)
        self.assertEqual(batch["revenue_at_risk"], 3000.0)
        self.assertEqual(batch["simulated_revenue_recovered"], 1500.0)
        self.assertEqual(batch["simulated_recovery_rate"], 0.5)
        self.assertEqual(batch["pct_of_oracle_revenue"], 0.85)
        self.assertEqual(batch["policy_override_rate"], 0.6567)
        self.assertEqual(batch["intervention_counts"], {"retry": 2, "payment_link": 1})
        self.assertEqual(sum(batch["intervention_counts"].values()), 3)

    def test_missing_comparison_summary_omits_oracle_fields(self):
        batch = build_batch_summary(
            decisions=[make_decision(final_action="retry")],
            transactions_processed=1,
            revenue_at_risk=1000.0,
            revenue_recovered=500.0,
            recovery_rate=0.5,
            passive_wait_baseline_pct_of_oracle_revenue=0.439,
            naive_historical_baseline_pct_of_oracle_revenue=0.590,
            full_comparison_summary=None,
        )
        self.assertNotIn("pct_of_oracle_revenue", batch)
        self.assertNotIn("policy_override_rate", batch)


@unittest.skipUnless(
    FULL_COMPARISON_JSON.exists() and HOLDOUT_PATH.exists() and MODEL_PATH.exists(),
    "requires the generated holdout/model/evaluation artifacts",
)
class TestBatchSummaryAgainstRealData(unittest.TestCase):
    """
    Cross-checks build_batch_summary() against the REAL, already-generated
    holdout run and outputs/full_policy_comparison.json. This does not
    recompute or alter the authoritative full-stack evaluation -- it only
    re-derives the same DecisionEngine/ActionExecutor pass app.py itself
    performs (read-only), and reads the JSON file as-is.
    """

    @classmethod
    def setUpClass(cls):
        import json

        import numpy as np
        import pandas as pd

        from action_executor import ActionExecutor
        from batch_runner import run_batch
        from decision_engine import DecisionEngine

        cls.holdout = pd.read_csv(HOLDOUT_PATH)
        engine = DecisionEngine(model_path=MODEL_PATH)
        executor = ActionExecutor()

        fields = [
            "transaction_id", "amount", "failure_reason",
            "prior_successful_payments", "prior_failed_payments",
            "prior_recovery_attempts", "hours_since_failure",
            "active_promise_to_pay", "contact_count_last_7_days", "is_quiet_hours",
        ]

        def row_to_txn(row):
            d = {k: row[k] for k in fields}
            d["active_promise_to_pay"] = bool(d["active_promise_to_pay"])
            d["is_quiet_hours"] = bool(d["is_quiet_hours"])
            for k in ["prior_successful_payments", "prior_failed_payments",
                      "prior_recovery_attempts", "contact_count_last_7_days"]:
                d[k] = int(d[k])
            return d

        transactions = [row_to_txn(row) for _, row in cls.holdout.iterrows()]
        cls.decisions = run_batch(transactions, engine)
        executions = [executor.execute_decision(d) for d in cls.decisions]

        rng = np.random.default_rng(42)
        revenue_at_risk = 0.0
        revenue_recovered = 0.0
        for (_, row), decision, execution in zip(cls.holdout.iterrows(), cls.decisions, executions):
            draw = rng.random()
            revenue_at_risk += float(row["amount"])
            if decision.status == "decided" and decision.final_action is not None:
                true_prob = float(row[f"true_prob_{decision.final_action}"])
                if execution.status == "executed" and draw < true_prob:
                    revenue_recovered += float(row["amount"])

        cls.revenue_at_risk = revenue_at_risk
        cls.revenue_recovered = revenue_recovered
        cls.recovery_rate = revenue_recovered / revenue_at_risk

        with open(FULL_COMPARISON_JSON) as f:
            cls.comparison_summary = json.load(f)

        cls.batch = build_batch_summary(
            decisions=cls.decisions,
            transactions_processed=len(cls.holdout),
            revenue_at_risk=cls.revenue_at_risk,
            revenue_recovered=cls.revenue_recovered,
            recovery_rate=cls.recovery_rate,
            passive_wait_baseline_pct_of_oracle_revenue=0.439,
            naive_historical_baseline_pct_of_oracle_revenue=0.590,
            full_comparison_summary=cls.comparison_summary,
        )

    def test_transactions_processed_is_300(self):
        self.assertEqual(self.batch["transactions_processed"], 300)

    def test_intervention_counts_sum_to_transactions_processed(self):
        self.assertEqual(
            sum(self.batch["intervention_counts"].values()),
            self.batch["transactions_processed"],
        )

    def test_pct_of_oracle_revenue_matches_json(self):
        self.assertEqual(
            self.batch["pct_of_oracle_revenue"],
            self.comparison_summary["pct_of_oracle_revenue"],
        )

    def test_policy_override_rate_matches_json(self):
        self.assertEqual(
            self.batch["policy_override_rate"],
            self.comparison_summary["policy_override_rate"],
        )

    def test_simulated_values_match_seeded_dashboard_computation(self):
        self.assertAlmostEqual(
            self.batch["simulated_revenue_recovered"], round(self.revenue_recovered, 2), places=2
        )
        self.assertAlmostEqual(
            self.batch["simulated_recovery_rate"], round(self.recovery_rate, 4), places=4
        )
        self.assertAlmostEqual(
            self.batch["revenue_at_risk"], round(self.revenue_at_risk, 2), places=2
        )


if __name__ == "__main__":
    unittest.main()
