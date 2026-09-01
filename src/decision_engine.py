"""
decision_engine.py

Turns one failed transaction into a bounded, auditable recovery decision:

    validate -> score all 5 actions (ML) -> apply policy -> select -> audit

The ML model proposes a recovery-probability score for every action; the
PolicyEngine independently decides which of those are actually allowed.
final_action is always chosen from the eligible set, never directly from
the raw model output — raw ML scores and policy eligibility are combined,
not conflated (see decide()). If the policy engine rejects every action,
the decision is held for manual review instead of forcing an unsafe
fallback.

Idempotency: DecisionEngine keeps an in-memory transaction_id -> result
cache. Repeating the same transaction_id returns the original decision
(marked duplicate=True) instead of recomputing it. This is a PROTOTYPE
mechanism only — it lives in process memory and does NOT persist across
restarts or across multiple processes. A production deployment would
back this with a persistent store (e.g. a database row or cache keyed by
transaction_id with a durability guarantee).
"""

import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from features import ACTIONS, add_action, to_model_matrix
from policy_engine import HIGH_VALUE_REVIEW_THRESHOLD, PolicyEngine
from schemas import DecisionResult, Transaction, ValidationError

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "recovery_model.joblib"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionEngine:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, policy_engine: PolicyEngine = None, model=None):
        # `model` lets tests inject a stub instead of the real trained
        # model, so policy/override behavior can be tested deterministically
        # without depending on what the real model happens to predict.
        self.model = model if model is not None else joblib.load(model_path)
        self.policy_engine = policy_engine or PolicyEngine()
        self._decisions_by_txn_id = {}

    def score_all_actions(self, txn: Transaction) -> dict:
        """Scores all 5 actions with the model, unfiltered by policy. This
        is the raw ML view — see module docstring on why it's kept
        separate from eligibility."""
        row = pd.DataFrame([txn.to_dict()])
        scores = {}
        for action in ACTIONS:
            X = to_model_matrix(add_action(row, action))
            scores[action] = float(self.model.predict_proba(X)[0, 1])
        return scores

    def build_recommendation_context(self, transaction: dict) -> dict:
        """
        Structured context for a future LLM/agent layer: the transaction,
        every action's ML prediction and expected value, and per-action
        eligibility with reasons. Whatever the future agent recommends
        must still be checked against policy_constraints here — it is
        never given authority to bypass them.
        """
        txn = Transaction.from_dict(transaction)
        ml_scores = self.score_all_actions(txn)
        policy_checks = self.policy_engine.evaluate_all(txn)
        return {
            "transaction": txn.to_dict(),
            "candidate_actions": list(ACTIONS),
            "ml_predictions": ml_scores,
            "expected_values": {a: txn.amount * p for a, p in ml_scores.items()},
            "policy_constraints": {c.action: c.to_dict() for c in policy_checks},
        }

    def decide(self, transaction: dict) -> DecisionResult:
        txn_id = transaction.get("transaction_id") if isinstance(transaction, dict) else None
        if txn_id in self._decisions_by_txn_id:
            return replace(self._decisions_by_txn_id[txn_id], duplicate=True)

        try:
            txn = Transaction.from_dict(transaction)
        except ValidationError as e:
            return DecisionResult(
                transaction_id=txn_id or "UNKNOWN",
                timestamp=_now_iso(),
                status="invalid_transaction",
                errors=e.errors,
            )

        ml_scores = self.score_all_actions(txn)
        expected_values = {a: txn.amount * p for a, p in ml_scores.items()}

        policy_checks = self.policy_engine.evaluate_all(txn)
        eligible_actions = [c.action for c in policy_checks if c.allowed]
        rejected_actions = [{"action": c.action, "reason": c.reason} for c in policy_checks if not c.allowed]

        # Tie-breaks for both max() calls below fall back to ACTIONS order
        # (retry, payment_link, reminder, escalate, wait) since ml_scores
        # and eligible_actions are both built by iterating ACTIONS in
        # order — deterministic and reproducible.
        model_recommendation = max(ml_scores, key=ml_scores.get)

        if eligible_actions:
            final_action = max(eligible_actions, key=lambda a: ml_scores[a])
            status = "decided"
        else:
            final_action = None
            status = "held_for_manual_review"

        policy_override = final_action != model_recommendation
        override_reason = None
        if policy_override:
            if final_action is None:
                override_reason = "All candidate actions were blocked by policy; held for manual review."
            else:
                rejected_lookup = {r["action"]: r["reason"] for r in rejected_actions}
                # model_recommendation will always be in rejected_lookup here
                # (if it were eligible, it would also be the top eligible
                # action, and final_action would equal it). Kept defensive.
                override_reason = rejected_lookup.get(
                    model_recommendation,
                    f"Model recommendation '{model_recommendation}' was not the top eligible action.",
                )

        is_high_value = self.policy_engine.is_high_value(txn)

        result = DecisionResult(
            transaction_id=txn.transaction_id,
            timestamp=_now_iso(),
            status=status,
            amount=txn.amount,
            failure_reason=txn.failure_reason,
            ml_scores=ml_scores,
            expected_recovery_values=expected_values,
            policy_checks=[c.to_dict() for c in policy_checks],
            eligible_actions=eligible_actions,
            rejected_actions=rejected_actions,
            model_recommendation=model_recommendation,
            final_action=final_action,
            policy_override=policy_override,
            override_reason=override_reason,
            flagged_for_review=is_high_value,
            flagged_for_review_reason=(
                f"Advisory only: amount {txn.amount:.2f} is at/above the prototype high-value "
                f"review threshold of {HIGH_VALUE_REVIEW_THRESHOLD:.2f} (does not affect "
                "eligibility or final_action)."
            )
            if is_high_value
            else None,
        )

        self._decisions_by_txn_id[txn.transaction_id] = result
        return result
