import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from action_executor import ActionExecutor
from decision_engine import DecisionEngine
from features import ACTIONS


def make_txn_dict(**overrides):
    defaults = dict(
        transaction_id="TEST-0001",
        amount=5000.0,
        failure_reason="card_declined",
        prior_successful_payments=3,
        prior_failed_payments=1,
        prior_recovery_attempts=1,
        hours_since_failure=10.0,
        active_promise_to_pay=False,
        contact_count_last_7_days=0,
        is_quiet_hours=False,
    )
    defaults.update(overrides)
    return defaults


class StubModel:
    """Deterministic stand-in for the trained model so override tests can
    control exactly which action the 'ML' prefers, independent of what
    the real trained model happens to predict for a given input."""

    def __init__(self, scores_by_action: dict):
        self.scores_by_action = scores_by_action

    def predict_proba(self, X):
        action = X["action"].iloc[0]
        p = self.scores_by_action[action]
        return np.array([[1 - p, p]])


class TestNormalTransaction(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()  # real trained model

    def test_all_five_actions_scored(self):
        result = self.engine.decide(make_txn_dict(transaction_id="NORM-1"))
        self.assertEqual(set(result.ml_scores.keys()), set(ACTIONS))
        self.assertEqual(set(result.expected_recovery_values.keys()), set(ACTIONS))

    def test_final_action_is_eligible_or_none(self):
        result = self.engine.decide(make_txn_dict(transaction_id="NORM-2"))
        if result.final_action is not None:
            self.assertIn(result.final_action, result.eligible_actions)
        else:
            self.assertEqual(result.status, "held_for_manual_review")

    def test_status_decided_for_normal_transaction(self):
        result = self.engine.decide(make_txn_dict(transaction_id="NORM-3"))
        self.assertEqual(result.status, "decided")
        self.assertIsNotNone(result.final_action)


class TestMalformedTransaction(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()

    def test_missing_amount_rejected(self):
        payload = make_txn_dict(transaction_id="BAD-1")
        del payload["amount"]
        result = self.engine.decide(payload)
        self.assertEqual(result.status, "invalid_transaction")
        self.assertTrue(any("amount" in e for e in result.errors))

    def test_negative_amount_rejected(self):
        result = self.engine.decide(make_txn_dict(transaction_id="BAD-2", amount=-100))
        self.assertEqual(result.status, "invalid_transaction")

    def test_unknown_failure_reason_rejected(self):
        result = self.engine.decide(make_txn_dict(transaction_id="BAD-3", failure_reason="alien_interference"))
        self.assertEqual(result.status, "invalid_transaction")

    def test_invalid_transaction_never_scores_or_executes(self):
        result = self.engine.decide(make_txn_dict(transaction_id="BAD-4", amount=-100))
        self.assertEqual(result.ml_scores, {})
        self.assertIsNone(result.final_action)


class TestWaitOverride(unittest.TestCase):
    def _make_stub_engine(self):
        stub = StubModel({"wait": 0.95, "retry": 0.4, "payment_link": 0.5, "reminder": 0.3, "escalate": 0.2})
        return DecisionEngine(model=stub)

    def test_wait_top_pick_blocked_without_ptp(self):
        engine = self._make_stub_engine()
        result = engine.decide(make_txn_dict(transaction_id="OVR-1", active_promise_to_pay=False))
        self.assertEqual(result.model_recommendation, "wait")
        self.assertNotEqual(result.final_action, "wait")
        self.assertTrue(result.policy_override)
        self.assertEqual(result.final_action, "payment_link")  # next-highest eligible score
        self.assertIn("promise-to-pay", result.override_reason)

    def test_wait_not_overridden_when_ptp_active(self):
        engine = self._make_stub_engine()
        result = engine.decide(make_txn_dict(transaction_id="OVR-2", active_promise_to_pay=True))
        self.assertEqual(result.final_action, "wait")
        self.assertFalse(result.policy_override)


class TestAllActionsBlocked(unittest.TestCase):
    def _blocked_payload(self, transaction_id):
        return make_txn_dict(
            transaction_id=transaction_id,
            active_promise_to_pay=False,  # blocks wait
            is_quiet_hours=True,  # blocks payment_link, reminder, escalate
            contact_count_last_7_days=10,  # also blocks contact actions
            prior_recovery_attempts=4,  # blocks retry (>= MAX_RETRY_ATTEMPTS)
        )

    def test_held_for_manual_review(self):
        engine = DecisionEngine()
        result = engine.decide(self._blocked_payload("BLOCK-1"))
        self.assertEqual(result.status, "held_for_manual_review")
        self.assertIsNone(result.final_action)
        self.assertEqual(result.eligible_actions, [])
        self.assertEqual(len(result.rejected_actions), 5)

    def test_blocked_action_never_reaches_executor(self):
        engine = DecisionEngine()
        executor = ActionExecutor()
        decision = engine.decide(self._blocked_payload("BLOCK-2"))
        execution = executor.execute_decision(decision)
        self.assertEqual(execution.status, "skipped_no_action")

    def test_audit_present_even_when_blocked(self):
        engine = DecisionEngine()
        decision = engine.decide(self._blocked_payload("BLOCK-3"))
        self.assertEqual(len(decision.ml_scores), 5)  # ML still scored everything
        self.assertEqual(len(decision.policy_checks), 5)
        self.assertTrue(all(not c["allowed"] for c in decision.policy_checks))


class TestIdempotency(unittest.TestCase):
    def test_repeated_transaction_id_returns_cached_decision(self):
        engine = DecisionEngine()
        payload = make_txn_dict(transaction_id="DUP-1")
        first = engine.decide(payload)
        second = engine.decide(payload)
        self.assertFalse(first.duplicate)
        self.assertTrue(second.duplicate)
        self.assertEqual(first.final_action, second.final_action)
        self.assertEqual(first.timestamp, second.timestamp)


class TestRecommendationContext(unittest.TestCase):
    def test_context_has_expected_keys(self):
        engine = DecisionEngine()
        ctx = engine.build_recommendation_context(make_txn_dict(transaction_id="CTX-1"))
        for key in ["transaction", "candidate_actions", "ml_predictions", "expected_values", "policy_constraints"]:
            self.assertIn(key, ctx)
        self.assertEqual(set(ctx["policy_constraints"].keys()), set(ACTIONS))


if __name__ == "__main__":
    unittest.main()
