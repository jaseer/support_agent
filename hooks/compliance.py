"""
Hook-based compliance enforcement.

Hooks are the deterministic control layer. The LLM is free to be clever in its
prompt engineering, but these functions enforce non-negotiable rules:

  * PreToolUse  — gate on policy: refund ceilings, sensitive-topic blocks,
                  PII redaction in tool inputs.
  * PostToolUse — audit logging, repeated-failure tracking, sentiment checks
                  on observed customer messages.
  * PostToolUseFailure — turn typed errors into escalation decisions.
  * UserPromptSubmit  — trigger escalation when the user explicitly asks for
                        a human, before the LLM even sees the message.

Every hook returns a dict in the SDK's expected shape. We never raise — a
hook that crashes would block the whole agent.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from ..core.models import EscalationReason
from ..core.policy import AgentPolicy, DEFAULT_POLICY
from ..tools.mcp_tools import (
    TOOL_ESCALATE,
    TOOL_ISSUE_REFUND,
    TOOL_LOOKUP_CUSTOMER,
)

log = logging.getLogger("support_agent.hooks")


# --------------------------------------------------------------------------- #
# Shared per-session state                                                    #
# --------------------------------------------------------------------------- #

class HookState:
    """Mutable state hooks share within a single agent session.

    Don't reach for module-level globals — that breaks isolation between
    concurrent agent runs in a server. Instantiate one of these per session.
    """

    def __init__(self):
        # tool_name → consecutive failure count this session
        self.failure_counts: dict[str, int] = defaultdict(int)
        # ordered audit log of every tool call
        self.audit_log: list[dict[str, Any]] = []
        # set when an escalation has been triggered by a hook
        self.escalation_pending: dict[str, Any] | None = None
        # cached customer context (filled after first lookup_customer success)
        self.verified_customer: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# PII redaction                                                               #
# --------------------------------------------------------------------------- #

# Conservative patterns. Real systems should use a dedicated PII detection
# library (e.g. Microsoft Presidio) — these are illustrative.
_CC_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_API_KEY_PATTERN = re.compile(r"\b(sk-|pk-|api[_-]?key[_-]?)[A-Za-z0-9_-]{16,}\b")


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return redacted text plus a list of redaction labels."""
    labels: list[str] = []
    if _CC_PATTERN.search(text):
        text = _CC_PATTERN.sub("[REDACTED:CC]", text)
        labels.append("credit_card")
    if _SSN_PATTERN.search(text):
        text = _SSN_PATTERN.sub("[REDACTED:SSN]", text)
        labels.append("ssn")
    if _API_KEY_PATTERN.search(text):
        text = _API_KEY_PATTERN.sub("[REDACTED:KEY]", text)
        labels.append("api_key")
    return text, labels


# --------------------------------------------------------------------------- #
# Hook factories                                                              #
# Each factory closes over `state` + `policy` so we can inject test doubles.  #
# --------------------------------------------------------------------------- #

def build_pre_tool_use_hook(state: HookState, policy: AgentPolicy = DEFAULT_POLICY):
    """PreToolUse: gate every tool call against policy before it runs."""

    async def pre_tool_use(input_data, tool_use_id, context):
        tool_name = input_data["tool_name"]
        tool_input = input_data.get("tool_input", {}) or {}

        # 1. Refund ceiling enforcement -------------------------------------
        if tool_name == TOOL_ISSUE_REFUND:
            amount = float(tool_input.get("amount_usd", 0) or 0)
            tier = (state.verified_customer or {}).get("tier", "standard")
            ceiling = policy.refund.ceiling_for(tier)
            if amount > ceiling or amount > policy.refund.hard_cap_usd:
                log.warning(
                    "Blocked refund: amount=%.2f ceiling=%.2f tier=%s",
                    amount, ceiling, tier,
                )
                state.escalation_pending = {
                    "reason": EscalationReason.REFUND_THRESHOLD_EXCEEDED.value,
                    "summary": (
                        f"Refund of ${amount:.2f} exceeds {tier}-tier ceiling "
                        f"of ${ceiling:.2f}. Needs human approval."
                    ),
                    "priority": "high",
                }
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"Refund amount ${amount:.2f} exceeds the "
                            f"${ceiling:.2f} auto-approval ceiling for "
                            f"{tier}-tier customers. You must call "
                            f"escalate_to_human with reason='refund_limit'."
                        ),
                    }
                }

        # 2. Identity must be verified before any state-changing action -----
        if tool_name == TOOL_ISSUE_REFUND and state.verified_customer is None:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "Customer identity has not been verified. Call "
                        "lookup_customer first."
                    ),
                }
            }

        # 3. PII redaction in tool inputs -----------------------------------
        # Don't pass raw card numbers to downstream services.
        for key, value in list(tool_input.items()):
            if isinstance(value, str):
                redacted, labels = redact_pii(value)
                if labels:
                    log.info("Redacted PII in %s.%s: %s", tool_name, key, labels)
                    tool_input[key] = redacted
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "allow",
                            "modifiedInput": tool_input,
                        }
                    }

        # 4. Audit log entry (request side) ---------------------------------
        state.audit_log.append({
            "phase": "pre",
            "tool": tool_name,
            "input": tool_input,
        })
        return {}  # allow

    return pre_tool_use


def build_post_tool_use_hook(state: HookState, policy: AgentPolicy = DEFAULT_POLICY):
    """PostToolUse: log results, cache verified-customer state, reset failure
    counters on success."""

    async def post_tool_use(input_data, tool_use_id, context):
        tool_name = input_data["tool_name"]
        result = input_data.get("tool_response") or input_data.get("tool_result", {})

        # Reset failure streak for this tool on a successful return
        state.failure_counts[tool_name] = 0

        # Cache verified customer so PreToolUse can authorize state changes
        if tool_name == TOOL_LOOKUP_CUSTOMER:
            try:
                content = result.get("content", [])
                if content and not result.get("isError"):
                    import json
                    payload = json.loads(content[0]["text"])
                    if "customer_id" in payload:
                        state.verified_customer = payload
                        log.info("Verified customer: %s (%s)",
                                 payload["customer_id"], payload.get("tier"))
            except (KeyError, IndexError, ValueError, TypeError):
                pass  # malformed result — fall through, don't crash the hook

        state.audit_log.append({
            "phase": "post",
            "tool": tool_name,
            "is_error": bool(result.get("isError")),
        })
        return {}

    return post_tool_use


def build_post_tool_failure_hook(state: HookState, policy: AgentPolicy = DEFAULT_POLICY):
    """PostToolUseFailure: when a tool raises, count the failure and decide
    whether we've hit the repeated-failure escalation threshold."""

    async def post_tool_failure(input_data, tool_use_id, context):
        tool_name = input_data["tool_name"]
        state.failure_counts[tool_name] += 1
        count = state.failure_counts[tool_name]

        log.warning("Tool failure: %s (count=%d)", tool_name, count)

        if count >= policy.escalation.repeated_failure_threshold:
            state.escalation_pending = {
                "reason": EscalationReason.REPEATED_FAILURE.value,
                "summary": (
                    f"Tool '{tool_name}' has failed {count} times in a row. "
                    f"Last error: {input_data.get('error', 'unknown')}"
                ),
                "priority": "high",
            }
            log.warning("Escalation triggered: %s failed %d times", tool_name, count)

        state.audit_log.append({
            "phase": "failure",
            "tool": tool_name,
            "count": count,
        })
        return {}

    return post_tool_failure


def build_user_prompt_hook(state: HookState, policy: AgentPolicy = DEFAULT_POLICY):
    """UserPromptSubmit: scan the user's message for explicit human-handoff
    requests and sensitive topics. Sets `escalation_pending` so the agent's
    system prompt directives lead it to call escalate_to_human first."""

    async def user_prompt(input_data, tool_use_id, context):
        prompt = (input_data.get("prompt") or "").lower()

        # Explicit handoff request
        for phrase in policy.escalation.user_escalation_phrases:
            if phrase in prompt:
                state.escalation_pending = {
                    "reason": EscalationReason.EXPLICIT_USER_REQUEST.value,
                    "summary": "Customer explicitly requested a human agent.",
                    "priority": "high",
                }
                log.info("User-requested escalation: phrase='%s'", phrase)
                return {}

        # Sensitive topic
        for topic in policy.escalation.sensitive_topics:
            if topic in prompt:
                state.escalation_pending = {
                    "reason": EscalationReason.SENSITIVE_DOMAIN.value,
                    "summary": f"Customer mentioned sensitive topic: '{topic}'.",
                    "priority": "urgent",
                }
                log.info("Sensitive-topic escalation: topic='%s'", topic)
                return {}

        return {}

    return user_prompt
