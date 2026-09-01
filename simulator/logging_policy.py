"""
logging_policy.py

Simulates how the business ACTUALLY behaved historically, before RecoverIQ
existed. This is deliberately simple and does NOT look at amount, customer
value, or failure_reason the way an optimal policy would. That's the point:
it creates realistic selection bias in the logged data, e.g. payment_link
is almost never tried on insufficient_funds cases historically, so a model
trained naively might never learn that combination well. Any honest writeup
should name this as a limitation of the logged data.

This is what generates your TRAINING data (Day 2). It is intentionally
suboptimal — a naive escalating-contact heuristic, roughly what an
unsophisticated ops team would do.
"""

import numpy as np

FROM_reward = None  # not used directly; kept for clarity of separation


def choose_logged_action(txn: dict, rng: np.random.Generator) -> str:
    attempts = txn["prior_recovery_attempts"]
    active_ptp = txn["active_promise_to_pay"]

    if active_ptp:
        # Even a naive team usually doesn't hammer someone who already
        # promised to pay -- but only ~70% of the time (real ops is messy).
        if rng.random() < 0.7:
            return "wait"

    if attempts == 0:
        return "retry"
    elif attempts == 1:
        return "reminder"
    elif attempts == 2:
        # small chance ops tries a payment link at this stage
        return "payment_link" if rng.random() < 0.3 else "reminder"
    else:
        return "escalate" if rng.random() < 0.5 else "reminder"
