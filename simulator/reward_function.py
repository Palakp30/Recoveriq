"""
reward_function.py

HIDDEN GROUND TRUTH — this file defines the TRUE probability that a given
recovery action succeeds for a given transaction. This is the simulator's
private knowledge. It is used to:
  1. Generate realized outcomes for the biased historical log (training data)
  2. Generate the full counterfactual matrix for the holdout set (eval only)

The ML model in Day 2 NEVER sees this file's logic. It only sees logged
(transaction, action, outcome) triples produced by logging_policy.py.

Design principle: each action has a different "profile" of which features
help it, so no single action dominates uniformly — this is what makes the
decision problem genuinely non-trivial rather than "always pick action X".
"""

import numpy as np

ACTIONS = ["retry", "payment_link", "reminder", "escalate", "wait"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def true_recovery_probability(txn: dict, action: str, rng: np.random.Generator) -> float:
    """
    Returns the TRUE probability [0,1] that `action` recovers `txn`.
    Includes an irreducible noise term (drawn once per call site is fine —
    the point is the underlying process is not perfectly deterministic,
    so no model should ever hit ~1.0 AUC and you should say so explicitly
    in your README).
    """
    amount = txn["amount"]
    log_amount = np.log1p(amount)
    prior_success = txn["prior_successful_payments"]
    prior_fail = txn["prior_failed_payments"]
    prior_recovery_attempts = txn["prior_recovery_attempts"]
    hours_since_failure = txn["hours_since_failure"]
    active_ptp = txn["active_promise_to_pay"]
    contact_count_7d = txn["contact_count_last_7_days"]
    failure_reason = txn["failure_reason"]
    quiet_hours = txn["is_quiet_hours"]

    # Small irreducible noise so the ceiling is realistic, never 100%.
    noise = rng.normal(0, 0.15)

    if action == "retry":
        # Retry works best for transient/technical failures, worse for
        # structural ones (insufficient funds rarely fixes itself instantly).
        base = -0.3
        if failure_reason in ("network_error", "bank_timeout"):
            base += 1.4
        if failure_reason == "insufficient_funds":
            base -= 0.6
        base += 0.15 * min(prior_success / 10, 1.5)  # loyal customers retry-succeed more
        base -= 0.35 * prior_recovery_attempts  # diminishing returns on repeated retries
        base -= 0.02 * hours_since_failure  # freshness matters for retries

    elif action == "payment_link":
        # Payment link is best for card/method failures, and scales
        # slightly better with higher amounts (customer more motivated).
        base = -0.5
        if failure_reason in ("card_declined", "expired_card"):
            base += 1.2
        base += 0.25 * min(log_amount / 10, 1.0)
        base -= 0.25 * prior_recovery_attempts
        if active_ptp:
            base += 0.4  # customer already engaged, link converts well

    elif action == "reminder":
        # Reminder is a low-cost, low-yield action. Strong diminishing
        # returns — the third reminder barely helps and can annoy.
        base = -0.4
        base += 0.3 * min(prior_success / 8, 1.2)
        base -= 0.5 * prior_recovery_attempts
        base -= 0.3 * contact_count_7d
        if quiet_hours:
            base -= 0.5  # reminders sent at bad times underperform

    elif action == "escalate":
        # Escalation (human agent contact) only pays off for high-value
        # or high-loyalty customers — expensive action, must be selective.
        base = -1.0
        base += 0.4 * min(log_amount / 8, 1.6)
        base += 0.3 * min(prior_success / 10, 1.2)
        base -= 0.2 * prior_fail

    elif action == "wait":
        # "Wait" is a deliberate no-contact action. It "succeeds" only in
        # the sense that an active promise-to-pay resolves on its own.
        base = -1.5
        if active_ptp:
            base += 2.2
        base -= 0.1 * hours_since_failure / 24

    else:
        raise ValueError(f"Unknown action: {action}")

    prob = _sigmoid(base + noise)
    return float(np.clip(prob, 0.01, 0.97))


def full_counterfactual_matrix(txn: dict, rng: np.random.Generator) -> dict:
    """Returns TRUE recovery probability for ALL actions for one transaction.
    ONLY used for holdout evaluation — never fed to the model or agent."""
    return {a: true_recovery_probability(txn, a, rng) for a in ACTIONS}
