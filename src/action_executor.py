"""
action_executor.py

Simulated execution of a policy-approved recovery action. This is NOT
real payment processing, messaging, or gateway integration — it only
records that "action X would have been executed for transaction Y" so
the rest of the system has something concrete to show. No real customer
contact happens here.

SECURITY / ARCHITECTURE NOTE (Day 3 hardening):
execute() does not accept a bare action string. It only accepts a
PolicyCheckResult (schemas.py) — the exact object PolicyEngine.check_action()
produces. This makes "prove you're policy-approved" a precondition for the
call itself, so a future caller (an LLM agent included) cannot execute an
action just by passing a string; it must go through something shaped like
a policy decision. ActionExecutor still does not know or re-derive *why*
an action is or isn't allowed — that stays entirely inside PolicyEngine.
The one exception is is_known_action(), imported (not reimplemented) from
policy_engine.py, which is a fixed-list membership check, not an
eligibility rule.

Idempotent per transaction_id: executing the same transaction_id twice
(regardless of what is requested the second time) returns the original
execution result instead of simulating again. Like DecisionEngine's
cache, this is an in-memory guard only — it does not persist across
process restarts and is not shared across processes.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent))
from policy_engine import is_known_action

from schemas import DecisionResult, ExecutionResult, PolicyCheckResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionExecutor:
    def __init__(self):
        self._executed_by_txn_id = {}

    def execute(self, transaction_id: str, policy_check: Optional[PolicyCheckResult]) -> ExecutionResult:
        """
        policy_check is either None (no action was approved for this
        transaction — e.g. held_for_manual_review) or a PolicyCheckResult
        obtained from PolicyEngine. Only allowed=True on a known action
        actually executes; everything else is rejected without running.
        """
        if transaction_id in self._executed_by_txn_id:
            previous = self._executed_by_txn_id[transaction_id]
            return ExecutionResult(
                transaction_id=transaction_id,
                action=previous.action,
                status="skipped_duplicate",
                simulation=True,
                timestamp=_now_iso(),
            )

        if policy_check is None:
            result = ExecutionResult(
                transaction_id=transaction_id,
                action=None,
                status="skipped_no_action",
                simulation=True,
                timestamp=_now_iso(),
            )
        elif not is_known_action(policy_check.action):
            result = ExecutionResult(
                transaction_id=transaction_id,
                action=policy_check.action,
                status="rejected_unknown_action",
                simulation=True,
                timestamp=_now_iso(),
            )
        elif not policy_check.allowed:
            result = ExecutionResult(
                transaction_id=transaction_id,
                action=policy_check.action,
                status="rejected_policy_not_approved",
                simulation=True,
                timestamp=_now_iso(),
            )
        else:
            result = ExecutionResult(
                transaction_id=transaction_id,
                action=policy_check.action,
                status="executed",
                simulation=True,
                timestamp=_now_iso(),
            )

        self._executed_by_txn_id[transaction_id] = result
        return result

    def execute_decision(self, decision: DecisionResult) -> ExecutionResult:
        """Convenience wrapper: execute the final_action a DecisionEngine
        already approved. Reconstructs the real PolicyCheckResult that
        PolicyEngine produced for final_action from decision.policy_checks
        (rather than trusting the string final_action on its own), so this
        goes through the same gate as any other caller."""
        if decision.final_action is None:
            return self.execute(decision.transaction_id, None)

        check = next(
            (c for c in decision.policy_checks if c["action"] == decision.final_action),
            None,
        )
        approved = PolicyCheckResult(**check) if check else None
        return self.execute(decision.transaction_id, approved)
