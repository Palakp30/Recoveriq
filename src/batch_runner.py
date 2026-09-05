"""
batch_runner.py

Runs the live decision path over a batch of transactions. Pure
orchestration -- no new scoring or policy logic, just repeated calls to
DecisionEngine.decide(). Reused by app.py (Day 4) and by any future batch
evaluation script (e.g. a replacement for the ad hoc holdout-evaluation
scripts), so there is exactly one place that loops decisions over a list.

Note: takes `engine` as an explicit parameter rather than constructing one
internally. This lets a caller build a single DecisionEngine once (e.g.
behind st.cache_resource, so the model is loaded from disk exactly once)
and reuse it across many run_batch() calls, instead of every call
reloading the joblib model and starting a fresh idempotency cache.
"""

from typing import List

from decision_engine import DecisionEngine
from schemas import DecisionResult


def run_batch(transactions: List[dict], engine: DecisionEngine) -> List[DecisionResult]:
    """Calls engine.decide() once per transaction dict, in order."""
    return [engine.decide(txn) for txn in transactions]
