import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from policy_engine import (
    HIGH_VALUE_REVIEW_THRESHOLD,
    MAX_CONTACTS_7_DAYS,
    MAX_RETRY_ATTEMPTS,
    MIN_ATTEMPTS_BEFORE_ESCALATE,
    PolicyEngine,
)
from schemas import Transaction


def make_txn(**overrides):
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
    return Transaction(**defaults)


class TestWaitEligibility(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_wait_blocked_without_active_ptp(self):
        result = self.engine.check_action(make_txn(active_promise_to_pay=False), "wait")
        self.assertFalse(result.allowed)
        self.assertIn("promise-to-pay", result.reason)

    def test_wait_allowed_with_active_ptp(self):
        result = self.engine.check_action(make_txn(active_promise_to_pay=True), "wait")
        self.assertTrue(result.allowed)
        self.assertIsNone(result.reason)


class TestQuietHours(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_quiet_hours_blocks_contact_actions(self):
        txn = make_txn(is_quiet_hours=True)
        for action in ["payment_link", "reminder", "escalate"]:
            result = self.engine.check_action(txn, action)
            self.assertFalse(result.allowed, f"{action} should be blocked in quiet hours")
            self.assertIn("quiet hours", result.reason)

    def test_quiet_hours_does_not_block_non_contact_actions(self):
        txn = make_txn(is_quiet_hours=True, active_promise_to_pay=True)
        for action in ["retry", "wait"]:
            result = self.engine.check_action(txn, action)
            self.assertTrue(result.allowed, f"{action} should not be blocked in quiet hours")


class TestContactFatigue(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_allowed_just_below_threshold(self):
        txn = make_txn(contact_count_last_7_days=MAX_CONTACTS_7_DAYS - 1)
        self.assertTrue(self.engine.check_action(txn, "reminder").allowed)

    def test_blocked_at_threshold(self):
        txn = make_txn(contact_count_last_7_days=MAX_CONTACTS_7_DAYS)
        result = self.engine.check_action(txn, "reminder")
        self.assertFalse(result.allowed)
        self.assertIn("contact limit", result.reason.lower())

    def test_non_contact_action_unaffected_by_contact_fatigue(self):
        txn = make_txn(contact_count_last_7_days=MAX_CONTACTS_7_DAYS + 5)
        self.assertTrue(self.engine.check_action(txn, "retry").allowed)


class TestRetryCap(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_allowed_just_below_cap(self):
        txn = make_txn(prior_recovery_attempts=MAX_RETRY_ATTEMPTS - 1)
        self.assertTrue(self.engine.check_action(txn, "retry").allowed)

    def test_blocked_at_cap(self):
        txn = make_txn(prior_recovery_attempts=MAX_RETRY_ATTEMPTS)
        result = self.engine.check_action(txn, "retry")
        self.assertFalse(result.allowed)
        self.assertIn("cap", result.reason.lower())


class TestEscalationMinimum(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_blocked_with_zero_attempts(self):
        txn = make_txn(prior_recovery_attempts=0)
        self.assertFalse(self.engine.check_action(txn, "escalate").allowed)

    def test_allowed_with_min_attempts(self):
        txn = make_txn(prior_recovery_attempts=MIN_ATTEMPTS_BEFORE_ESCALATE)
        self.assertTrue(self.engine.check_action(txn, "escalate").allowed)


class TestUnsupportedAction(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_unknown_action_rejected(self):
        result = self.engine.check_action(make_txn(), "refund")
        self.assertFalse(result.allowed)
        self.assertIn("Unsupported", result.reason)


class TestHighValueAdvisoryFlag(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine()

    def test_flag_set_above_threshold(self):
        txn = make_txn(amount=HIGH_VALUE_REVIEW_THRESHOLD + 1)
        self.assertTrue(self.engine.is_high_value(txn))

    def test_flag_not_set_below_threshold(self):
        txn = make_txn(amount=HIGH_VALUE_REVIEW_THRESHOLD - 1)
        self.assertFalse(self.engine.is_high_value(txn))

    def test_high_value_does_not_change_eligibility(self):
        # A high-value transaction with no other constraint triggered
        # must still be allowed — the flag is advisory only.
        txn = make_txn(amount=HIGH_VALUE_REVIEW_THRESHOLD + 1, active_promise_to_pay=False)
        result = self.engine.check_action(txn, "retry")
        self.assertTrue(result.allowed)


if __name__ == "__main__":
    unittest.main()
