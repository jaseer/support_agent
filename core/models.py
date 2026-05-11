"""
Domain models and structured errors for the customer support agent.

The error hierarchy is intentionally narrow: every failure category gets its own
exception class so PostToolUseFailure hooks can dispatch on type rather than
parsing strings out of error messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


# --------------------------------------------------------------------------- #
# Enums                                                                       #
# --------------------------------------------------------------------------- #

class TicketPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class EscalationReason(str, Enum):
    """Why we kicked a ticket up to a human.

    These map 1:1 to lines in the runbook a human agent picks up. Keep the set
    small — every reason should drive a different human workflow.
    """
    LOW_CONFIDENCE = "low_confidence"           # Agent isn't sure of the answer
    POLICY_VIOLATION_RISK = "policy_risk"       # Action would breach a guardrail
    EXPLICIT_USER_REQUEST = "user_requested"    # Customer asked for a human
    SENTIMENT_NEGATIVE = "sentiment_negative"   # Frustration / abuse detected
    REFUND_THRESHOLD_EXCEEDED = "refund_limit"  # Over auto-approval ceiling
    REPEATED_FAILURE = "repeated_failure"       # Same tool failed N times
    SENSITIVE_DOMAIN = "sensitive_domain"       # Legal / medical / billing dispute
    UNKNOWN_CUSTOMER = "unknown_customer"       # Identity check failed


# --------------------------------------------------------------------------- #
# Domain objects                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Customer:
    customer_id: str
    email: str
    name: str
    tier: str = "standard"           # "standard" | "pro" | "enterprise"
    lifetime_value_usd: float = 0.0


@dataclass
class Order:
    order_id: str
    customer_id: str
    total_usd: float
    status: str                      # "pending" | "shipped" | "delivered" | "cancelled"
    created_at: datetime


@dataclass
class Ticket:
    ticket_id: str = field(default_factory=lambda: f"T-{uuid4().hex[:8]}")
    customer_id: str | None = None
    subject: str = ""
    priority: TicketPriority = TicketPriority.NORMAL
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transcript: list[dict[str, Any]] = field(default_factory=list)
    escalation_reason: EscalationReason | None = None
    escalation_notes: str | None = None


# --------------------------------------------------------------------------- #
# Structured errors                                                           #
# --------------------------------------------------------------------------- #

class SupportAgentError(Exception):
    """Base class. Carries a stable `code` plus optional structured context.

    The agent's PostToolUseFailure hook reads `code` and `escalate` to decide
    whether to retry, fall back, or escalate to a human. Strings in `message`
    are for humans; never parse them in code.
    """
    code: str = "agent_error"
    escalate: bool = False           # Should this failure auto-escalate?
    retryable: bool = False          # Safe for the agent to retry?

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def to_tool_error(self) -> dict[str, Any]:
        """Serialize for return inside a tool's `content` block.

        Tool errors are returned as content (not raised) so Claude sees them
        and can adapt — e.g., try a different lookup or ask the user a
        clarifying question.
        """
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "escalate": self.escalate,
            "context": self.context,
        }


class CustomerNotFoundError(SupportAgentError):
    code = "customer_not_found"
    escalate = False                 # Ask for clarification, don't escalate
    retryable = False


class OrderNotFoundError(SupportAgentError):
    code = "order_not_found"
    escalate = False
    retryable = False


class AuthenticationError(SupportAgentError):
    code = "auth_failed"
    escalate = True                  # Identity check failed → human
    retryable = False


class PolicyViolationError(SupportAgentError):
    """Raised by hooks when a tool call would breach policy."""
    code = "policy_violation"
    escalate = True
    retryable = False


class RefundLimitExceededError(SupportAgentError):
    code = "refund_limit_exceeded"
    escalate = True
    retryable = False


class ExternalServiceError(SupportAgentError):
    """Transient downstream failure — usually safe to retry once."""
    code = "external_service_error"
    escalate = False
    retryable = True
