"""
Policy configuration. All escalation thresholds live here so they can be
tuned without touching agent or hook logic.

If you find yourself adding a `if customer.tier == ...` branch in a hook or
tool, add it here instead and reference it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RefundPolicy:
    # Auto-approval ceilings by tier (USD). Anything above → escalate.
    auto_approve_ceiling_usd: dict[str, float] = field(
        default_factory=lambda: {
            "standard": 50.0,
            "pro": 250.0,
            "enterprise": 1_000.0,
        }
    )
    # Hard cap: no refund larger than this auto-issued under any condition.
    hard_cap_usd: float = 1_000.0

    def ceiling_for(self, tier: str) -> float:
        return self.auto_approve_ceiling_usd.get(tier, 50.0)


@dataclass(frozen=True)
class EscalationPolicy:
    # Trigger escalation if the same tool fails this many times in a turn.
    repeated_failure_threshold: int = 3

    # Phrases that immediately escalate regardless of agent confidence.
    user_escalation_phrases: tuple[str, ...] = (
        "speak to a human", "talk to a person", "real person",
        "human agent", "manager", "supervisor",
        "this is unacceptable", "i'll sue", "lawyer",
    )

    # Topics that always require a human (regulatory, legal, safety).
    sensitive_topics: tuple[str, ...] = (
        "legal action", "lawsuit", "subpoena",
        "data breach", "gdpr deletion", "ccpa request",
        "medical advice", "injury", "harm",
    )

    # Sentiment threshold (negative score) above which we escalate.
    negative_sentiment_threshold: float = 0.7


@dataclass(frozen=True)
class AgentPolicy:
    refund: RefundPolicy = field(default_factory=RefundPolicy)
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)

    # Agent confidence below this triggers escalation rather than answering.
    min_answer_confidence: float = 0.6


# Singleton-style default. Tests and runtime overrides construct their own.
DEFAULT_POLICY = AgentPolicy()
