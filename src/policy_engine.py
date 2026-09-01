"""
policy_engine.py

Centralized, deterministic policy layer for RecoverIQ.

Within the live decision path (decision_engine.py, action_executor.py),
this is the ONLY place recovery-action eligibility rules live. The ML
model (models/recovery_model.joblib) estimates how likely an action is
to work; this engine decides whether that action is allowed to run at
all, for safety/business reasons. The ML model never bypasses these
checks — and neither will the future LLM agent: any action it proposes
must still be validated here (see is_known_action / check_action) before
it can become a final_action.

(Note: src/evaluate_policy.py's Day 2 masked_model_policy() contains its
own independent, frozen copy of the WAIT-eligibility rule, predating this
engine. It is intentionally left as-is to preserve Day 2's reproducible
results — it is not part of the live decision path and is not read by
DecisionEngine or ActionExecutor.)

All thresholds are module-level constants grouped in one place so they
can be tuned without hunting through multiple files. Each is documented
with where it comes from — none are invented just to look sophisticated.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from features import ACTIONS  # reuse the single canonical action list

from schemas import PolicyCheckResult, Transaction

# Which actions actually reach out to the customer. retry is a silent
# backend re-charge attempt; wait is no action at all — neither contacts
# the customer, so neither is subject to the quiet-hours / contact-fatigue
# rules below.
ACTION_METADATA = {
    "retry": {
        "requires_customer_contact": False,
        "description": "Silent backend retry of the payment charge.",
    },
    "payment_link": {
        "requires_customer_contact": True,
        "description": "Sends a new payment link to the customer.",
    },
    "reminder": {
        "requires_customer_contact": True,
        "description": "Sends a reminder message/notification to the customer.",
    },
    "escalate": {
        "requires_customer_contact": True,
        "description": "Routes to a human agent who contacts the customer directly.",
    },
    "wait": {
        "requires_customer_contact": False,
        "description": "No action taken; relies on an existing promise-to-pay resolving on its own.",
    },
}

# --- Configurable thresholds ---------------------------------------------

# Rule: contact fatigue. In the logged Day 1/Day 2 data, ~95% of
# transactions have contact_count_last_7_days <= 2, and the reward
# function itself penalizes "reminder" by -0.3 per contact in the last 7
# days (diminishing/negative returns). 3+ contacts blocks further
# customer-contact actions.
MAX_CONTACTS_7_DAYS = 3

# Rule: retry attempt cap. 4 is the simulator's own hard ceiling on
# prior_recovery_attempts (min(poisson(1.0), 4) in generate_data.py) —
# never exceeded in either dataset, not an arbitrary number. Beyond it,
# further silent retries are blocked.
MAX_RETRY_ATTEMPTS = 4

# Rule: escalation requires at least one prior attempt. escalate has the
# most negative base score in the reward function (most expensive
# action); a cheaper action should be tried before costly human
# escalation.
MIN_ATTEMPTS_BEFORE_ESCALATE = 1

# Rule: HIGH-VALUE REVIEW FLAG — ADVISORY ONLY, NEVER BLOCKING.
# ~95th percentile of the logged transaction amount distribution
# (empirically ~24,587 on the current Day 1 data), rounded to a readable
# number. This is a prototype heuristic for flagging a decision for human
# review, NOT a validated financial-risk threshold. It must never change
# eligible_actions or final_action.
HIGH_VALUE_REVIEW_THRESHOLD = 25000.0

CONTACT_ACTIONS = {a for a, meta in ACTION_METADATA.items() if meta["requires_customer_contact"]}


def is_known_action(action: str) -> bool:
    return action in ACTIONS


class PolicyEngine:
    """Stateless: every method takes the transaction/action it needs and
    returns an answer. No hidden state, trivial to unit test in isolation."""

    def check_action(self, txn: Transaction, action: str) -> PolicyCheckResult:
        if not is_known_action(action):
            return PolicyCheckResult(action=action, allowed=False, reason=f"Unsupported action: {action!r}.")

        if action == "wait" and not txn.active_promise_to_pay:
            return PolicyCheckResult(
                action=action,
                allowed=False,
                reason="WAIT requires an active promise-to-pay.",
            )

        requires_contact = ACTION_METADATA[action]["requires_customer_contact"]

        if requires_contact and txn.is_quiet_hours:
            return PolicyCheckResult(
                action=action,
                allowed=False,
                reason="Customer contact is restricted during quiet hours.",
            )

        if requires_contact and txn.contact_count_last_7_days >= MAX_CONTACTS_7_DAYS:
            return PolicyCheckResult(
                action=action,
                allowed=False,
                reason=(
                    f"Customer contact limit reached "
                    f"({txn.contact_count_last_7_days} >= {MAX_CONTACTS_7_DAYS} contacts in 7 days)."
                ),
            )

        if action == "retry" and txn.prior_recovery_attempts >= MAX_RETRY_ATTEMPTS:
            return PolicyCheckResult(
                action=action,
                allowed=False,
                reason=f"Retry attempt cap reached ({txn.prior_recovery_attempts} >= {MAX_RETRY_ATTEMPTS}).",
            )

        if action == "escalate" and txn.prior_recovery_attempts < MIN_ATTEMPTS_BEFORE_ESCALATE:
            return PolicyCheckResult(
                action=action,
                allowed=False,
                reason=(
                    f"Escalation requires at least {MIN_ATTEMPTS_BEFORE_ESCALATE} prior recovery "
                    f"attempt(s) ({txn.prior_recovery_attempts} so far)."
                ),
            )

        return PolicyCheckResult(action=action, allowed=True, reason=None)

    def evaluate_all(self, txn: Transaction) -> list:
        """PolicyCheckResult for each of the 5 known actions, in ACTIONS order."""
        return [self.check_action(txn, action) for action in ACTIONS]

    def is_high_value(self, txn: Transaction) -> bool:
        """Advisory only — callers must NOT use this to change eligibility."""
        return txn.amount >= HIGH_VALUE_REVIEW_THRESHOLD
