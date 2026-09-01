import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from action_executor import ActionExecutor
from features import ACTIONS
from schemas import PolicyCheckResult


class TestActionExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = ActionExecutor()

    def test_none_action_is_skipped_not_executed(self):
        result = self.executor.execute("TXN-NONE", None)
        self.assertEqual(result.status, "skipped_no_action")
        self.assertIsNone(result.action)
        self.assertTrue(result.simulation)

    def test_each_of_five_actions_executes_when_approved(self):
        for i, action in enumerate(ACTIONS):
            approved = PolicyCheckResult(action=action, allowed=True)
            result = self.executor.execute(f"TXN-{i}", approved)
            self.assertEqual(result.status, "executed")
            self.assertEqual(result.action, action)
            self.assertTrue(result.simulation)

    def test_duplicate_execution_is_skipped(self):
        first = self.executor.execute("TXN-DUP", PolicyCheckResult(action="payment_link", allowed=True))
        # Even a different, also-approved action requested the 2nd time
        # must not run -- the transaction_id is already settled.
        second = self.executor.execute("TXN-DUP", PolicyCheckResult(action="retry", allowed=True))
        self.assertEqual(first.status, "executed")
        self.assertEqual(second.status, "skipped_duplicate")
        self.assertEqual(second.action, "payment_link")  # reflects the ORIGINAL executed action

    def test_unknown_action_is_rejected_not_executed(self):
        bogus = PolicyCheckResult(action="refund_everything", allowed=True)  # lying about approval
        result = self.executor.execute("TXN-UNKNOWN", bogus)
        self.assertEqual(result.status, "rejected_unknown_action")
        self.assertNotEqual(result.status, "executed")

    def test_policy_blocked_action_cannot_execute(self):
        blocked = PolicyCheckResult(action="wait", allowed=False, reason="WAIT requires an active promise-to-pay.")
        result = self.executor.execute("TXN-BLOCKED", blocked)
        self.assertEqual(result.status, "rejected_policy_not_approved")
        self.assertNotEqual(result.status, "executed")


if __name__ == "__main__":
    unittest.main()
