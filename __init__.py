"""Customer support agent — public surface."""
from .agent import AgentResult, CustomerSupportAgent
from .core.models import (
    EscalationReason,
    PolicyViolationError,
    RefundLimitExceededError,
    SupportAgentError,
    Ticket,
    TicketPriority,
    TicketStatus,
)
from .core.policy import AgentPolicy, DEFAULT_POLICY
from .core.services import ServiceRegistry

__all__ = [
    "AgentResult",
    "AgentPolicy",
    "CustomerSupportAgent",
    "DEFAULT_POLICY",
    "EscalationReason",
    "PolicyViolationError",
    "RefundLimitExceededError",
    "ServiceRegistry",
    "SupportAgentError",
    "Ticket",
    "TicketPriority",
    "TicketStatus",
]
