"""
schemas.py

Plain dataclasses for the Day 3 decision engine. No pydantic — kept to the
stdlib so Day 3 adds zero new dependencies (pydantic/pytest are not
installed in this environment; unittest covers testing instead).

Transaction.from_dict() is the single place malformed input is rejected.
It never raises past DecisionEngine — DecisionEngine catches
ValidationError and turns it into a structured DecisionResult instead of
letting a bad payload reach feature construction or the model.
"""

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent / "simulator"))
from generate_data import FAILURE_REASONS  # reuse the simulator's own list, don't redeclare

REQUIRED_FIELDS = [
    "transaction_id",
    "amount",
    "failure_reason",
    "prior_successful_payments",
    "prior_failed_payments",
    "prior_recovery_attempts",
    "hours_since_failure",
    "active_promise_to_pay",
    "contact_count_last_7_days",
    "is_quiet_hours",
]

_NON_NEGATIVE_INT_FIELDS = [
    "prior_successful_payments",
    "prior_failed_payments",
    "prior_recovery_attempts",
    "contact_count_last_7_days",
]

_BOOLEAN_FIELDS = ["active_promise_to_pay", "is_quiet_hours"]


class ValidationError(Exception):
    """Raised by Transaction.from_dict on malformed input. Always caught
    inside DecisionEngine.decide() — never propagates to a caller."""

    def __init__(self, errors: list):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class Transaction:
    transaction_id: str
    amount: float
    failure_reason: str
    prior_successful_payments: int
    prior_failed_payments: int
    prior_recovery_attempts: int
    hours_since_failure: float
    active_promise_to_pay: bool
    contact_count_last_7_days: int
    is_quiet_hours: bool

    @classmethod
    def from_dict(cls, data: dict) -> "Transaction":
        if not isinstance(data, dict):
            raise ValidationError(["transaction payload must be a dict"])

        errors = [f"missing required field: {f}" for f in REQUIRED_FIELDS if data.get(f) is None]
        if errors:
            raise ValidationError(errors)

        txn_id = data["transaction_id"]
        if not isinstance(txn_id, str) or not txn_id.strip():
            errors.append("transaction_id must be a non-empty string")

        amount = data["amount"]
        if not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount <= 0:
            errors.append("amount must be a positive number")

        failure_reason = data["failure_reason"]
        if failure_reason not in FAILURE_REASONS:
            errors.append(f"unknown failure_reason: {failure_reason!r} (expected one of {FAILURE_REASONS})")

        for f in _NON_NEGATIVE_INT_FIELDS:
            v = data[f]
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                errors.append(f"{f} must be a non-negative integer")

        hours = data["hours_since_failure"]
        if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours < 0:
            errors.append("hours_since_failure must be a non-negative number")

        for f in _BOOLEAN_FIELDS:
            if not isinstance(data[f], bool):
                errors.append(f"{f} must be a boolean")

        if errors:
            raise ValidationError(errors)

        return cls(
            transaction_id=txn_id,
            amount=float(amount),
            failure_reason=failure_reason,
            prior_successful_payments=int(data["prior_successful_payments"]),
            prior_failed_payments=int(data["prior_failed_payments"]),
            prior_recovery_attempts=int(data["prior_recovery_attempts"]),
            hours_since_failure=float(hours),
            active_promise_to_pay=bool(data["active_promise_to_pay"]),
            contact_count_last_7_days=int(data["contact_count_last_7_days"]),
            is_quiet_hours=bool(data["is_quiet_hours"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyCheckResult:
    action: str
    allowed: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionResult:
    transaction_id: str
    timestamp: str
    # "decided" | "invalid_transaction" | "held_for_manual_review"
    status: str
    amount: Optional[float] = None
    failure_reason: Optional[str] = None
    ml_scores: dict = field(default_factory=dict)
    expected_recovery_values: dict = field(default_factory=dict)
    policy_checks: list = field(default_factory=list)
    eligible_actions: list = field(default_factory=list)
    rejected_actions: list = field(default_factory=list)
    model_recommendation: Optional[str] = None
    final_action: Optional[str] = None
    policy_override: bool = False
    override_reason: Optional[str] = None
    # Advisory-only flag — see policy_engine.HIGH_VALUE_REVIEW_THRESHOLD.
    # Must never influence eligible_actions or final_action.
    flagged_for_review: bool = False
    flagged_for_review_reason: Optional[str] = None
    errors: list = field(default_factory=list)
    duplicate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionResult:
    transaction_id: str
    action: Optional[str]
    # "executed" | "skipped_duplicate" | "skipped_no_action"
    status: str
    simulation: bool = True
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
