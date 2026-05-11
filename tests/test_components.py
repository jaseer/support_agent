"""
Test harness for the support agent.

Exercises the deterministic components — tools, hooks, errors, policy — using
fakes and direct calls. The agent's LLM-driven loop is tested separately with
a live API key (see `demo.py`).

Run:  python -m support_agent.tests.test_components
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from typing import Any

from ..core.models import (
    CustomerNotFoundError,
    EscalationReason,
    OrderNotFoundError,
    RefundLimitExceededError,
    SupportAgentError,
)
from ..core.policy import AgentPolicy, DEFAULT_POLICY
from ..core.services import CRMService, OrderService, RefundService, ServiceRegistry, TicketingService
from ..hooks.compliance import (
    HookState,
    build_post_tool_failure_hook,
    build_post_tool_use_hook,
    build_pre_tool_use_hook,
    build_user_prompt_hook,
    redact_pii,
)
from ..tools.mcp_tools import (
    TOOL_ESCALATE,
    TOOL_ISSUE_REFUND,
    TOOL_LOOKUP_CUSTOMER,
)


# --------------------------------------------------------------------------- #
# Lightweight test runner — no pytest dependency                              #
# --------------------------------------------------------------------------- #

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[tuple[str, str]] = []

    async def run(self, name: str, coro):
        try:
            await coro
            self.passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, f"AssertionError: {e}"))
            print(f"  FAIL  {name}: {e}")
        except Exception:
            self.failed += 1
            tb = traceback.format_exc()
            self.errors.append((name, tb))
            print(f"  FAIL  {name}\n{tb}")

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Results: {self.passed}/{total} passed")
        if self.failed:
            print(f"Failed: {self.failed}")
            return 1
        return 0


# --------------------------------------------------------------------------- #
# Test cases                                                                  #
# --------------------------------------------------------------------------- #

# ---- Errors -------------------------------------------------------------- #

async def test_error_serialization():
    err = RefundLimitExceededError(
        "amount too high",
        context={"amount_usd": 500.0, "ceiling_usd": 50.0},
    )
    payload = err.to_tool_error()
    assert payload["code"] == "refund_limit_exceeded"
    assert payload["escalate"] is True
    assert payload["retryable"] is False
    assert payload["context"]["amount_usd"] == 500.0


async def test_error_hierarchy():
    # All custom errors must inherit from SupportAgentError so a single
    # except clause in tools can catch them all.
    for cls in [CustomerNotFoundError, OrderNotFoundError, RefundLimitExceededError]:
        assert issubclass(cls, SupportAgentError), f"{cls} not in hierarchy"


# ---- Services ------------------------------------------------------------ #

async def test_crm_lookup_known_customer():
    crm = CRMService()
    customer = await crm.get_customer("C-1001")
    assert customer.tier == "enterprise"
    assert customer.email == "alice@example.com"


async def test_crm_lookup_unknown_raises():
    crm = CRMService()
    try:
        await crm.get_customer("C-DOES-NOT-EXIST")
        assert False, "Should have raised CustomerNotFoundError"
    except CustomerNotFoundError as e:
        assert e.context["customer_id"] == "C-DOES-NOT-EXIST"


async def test_crm_find_by_email():
    crm = CRMService()
    customer = await crm.find_by_email("BOB@example.com")  # case-insensitive
    assert customer.customer_id == "C-1002"


async def test_order_lookup():
    orders = OrderService()
    order = await orders.get_order("O-5001")
    assert order.total_usd == 1_249.00
    assert order.status == "delivered"


async def test_order_list_by_customer():
    orders = OrderService()
    alice_orders = await orders.list_for_customer("C-1001")
    assert len(alice_orders) == 2


# ---- Policy ------------------------------------------------------------- #

async def test_refund_ceiling_per_tier():
    policy = DEFAULT_POLICY
    assert policy.refund.ceiling_for("standard") == 50.0
    assert policy.refund.ceiling_for("pro") == 250.0
    assert policy.refund.ceiling_for("enterprise") == 1_000.0
    # Unknown tier defaults to standard ceiling — fail-closed
    assert policy.refund.ceiling_for("vip-unknown") == 50.0


# ---- PII redaction ------------------------------------------------------ #

async def test_redact_credit_card():
    text = "My card is 4111 1111 1111 1111 please charge it"
    redacted, labels = redact_pii(text)
    assert "4111" not in redacted
    assert "[REDACTED:CC]" in redacted
    assert "credit_card" in labels


async def test_redact_ssn():
    text = "SSN is 123-45-6789"
    redacted, labels = redact_pii(text)
    assert "123-45-6789" not in redacted
    assert "ssn" in labels


async def test_redact_passes_clean_text():
    text = "Hi, my order arrived broken"
    redacted, labels = redact_pii(text)
    assert redacted == text
    assert labels == []


# ---- PreToolUse hook --------------------------------------------------- #

async def test_pre_hook_blocks_oversized_refund():
    state = HookState()
    state.verified_customer = {"customer_id": "C-1002", "tier": "standard"}
    hook = build_pre_tool_use_hook(state)

    result = await hook(
        input_data={
            "tool_name": TOOL_ISSUE_REFUND,
            "tool_input": {"amount_usd": 500.0, "order_id": "O-5003", "customer_id": "C-1002", "reason": "test"},
        },
        tool_use_id="tool-1",
        context=None,
    )
    output = result.get("hookSpecificOutput", {})
    assert output.get("permissionDecision") == "deny"
    assert "ceiling" in output.get("permissionDecisionReason", "").lower()
    # And the escalation pending flag is set so the agent knows to escalate
    assert state.escalation_pending is not None
    assert state.escalation_pending["reason"] == EscalationReason.REFUND_THRESHOLD_EXCEEDED.value


async def test_pre_hook_allows_within_ceiling():
    state = HookState()
    state.verified_customer = {"customer_id": "C-1001", "tier": "enterprise"}
    hook = build_pre_tool_use_hook(state)

    result = await hook(
        input_data={
            "tool_name": TOOL_ISSUE_REFUND,
            "tool_input": {"amount_usd": 800.0, "order_id": "O-5001", "customer_id": "C-1001", "reason": "test"},
        },
        tool_use_id="tool-1",
        context=None,
    )
    # No deny output → allowed
    assert result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
    assert state.escalation_pending is None


async def test_pre_hook_blocks_refund_without_verification():
    state = HookState()  # no verified_customer
    hook = build_pre_tool_use_hook(state)

    result = await hook(
        input_data={
            "tool_name": TOOL_ISSUE_REFUND,
            "tool_input": {"amount_usd": 20.0, "order_id": "O-5003", "customer_id": "C-1002", "reason": "test"},
        },
        tool_use_id="tool-1",
        context=None,
    )
    output = result.get("hookSpecificOutput", {})
    assert output.get("permissionDecision") == "deny"
    assert "verified" in output.get("permissionDecisionReason", "").lower()


async def test_pre_hook_redacts_pii_in_input():
    state = HookState()
    hook = build_pre_tool_use_hook(state)
    result = await hook(
        input_data={
            "tool_name": TOOL_ESCALATE,
            "tool_input": {
                "ticket_id": "T-1",
                "reason": "user_requested",
                "summary": "Customer's card 4111 1111 1111 1111 was charged twice",
                "priority": "high",
            },
        },
        tool_use_id="tool-1",
        context=None,
    )
    output = result.get("hookSpecificOutput", {})
    # The hook should have allowed but modified the input
    assert output.get("permissionDecision") == "allow"
    modified = output.get("modifiedInput", {})
    assert "4111" not in modified.get("summary", "")
    assert "[REDACTED:CC]" in modified.get("summary", "")


# ---- PostToolUse hook -------------------------------------------------- #

async def test_post_hook_caches_verified_customer():
    state = HookState()
    hook = build_post_tool_use_hook(state)
    fake_result = {
        "content": [{"type": "text", "text": json.dumps({
            "customer_id": "C-1003",
            "name": "Carol Singh",
            "tier": "pro",
        })}],
        "isError": False,
    }
    await hook(
        input_data={"tool_name": TOOL_LOOKUP_CUSTOMER, "tool_response": fake_result},
        tool_use_id="t-1",
        context=None,
    )
    assert state.verified_customer is not None
    assert state.verified_customer["tier"] == "pro"


async def test_post_hook_resets_failure_count_on_success():
    state = HookState()
    state.failure_counts[TOOL_LOOKUP_CUSTOMER] = 2
    hook = build_post_tool_use_hook(state)
    await hook(
        input_data={
            "tool_name": TOOL_LOOKUP_CUSTOMER,
            "tool_response": {"content": [{"type": "text", "text": "{}"}], "isError": False},
        },
        tool_use_id="t-1",
        context=None,
    )
    assert state.failure_counts[TOOL_LOOKUP_CUSTOMER] == 0


# ---- Failure hook ------------------------------------------------------ #

async def test_failure_hook_triggers_escalation_after_threshold():
    state = HookState()
    hook = build_post_tool_failure_hook(state)
    # Threshold is 3 by default
    for i in range(3):
        await hook(
            input_data={"tool_name": TOOL_LOOKUP_CUSTOMER, "error": "timeout"},
            tool_use_id=f"t-{i}",
            context=None,
        )
    assert state.escalation_pending is not None
    assert state.escalation_pending["reason"] == EscalationReason.REPEATED_FAILURE.value


async def test_failure_hook_no_escalation_below_threshold():
    state = HookState()
    hook = build_post_tool_failure_hook(state)
    for i in range(2):
        await hook(
            input_data={"tool_name": TOOL_LOOKUP_CUSTOMER, "error": "timeout"},
            tool_use_id=f"t-{i}",
            context=None,
        )
    assert state.escalation_pending is None


# ---- User prompt hook -------------------------------------------------- #

async def test_user_prompt_hook_explicit_escalation():
    state = HookState()
    hook = build_user_prompt_hook(state)
    await hook(
        input_data={"prompt": "I want to speak to a human right now"},
        tool_use_id=None,
        context=None,
    )
    assert state.escalation_pending is not None
    assert state.escalation_pending["reason"] == EscalationReason.EXPLICIT_USER_REQUEST.value


async def test_user_prompt_hook_sensitive_topic():
    state = HookState()
    hook = build_user_prompt_hook(state)
    await hook(
        input_data={"prompt": "I'm considering legal action over this"},
        tool_use_id=None,
        context=None,
    )
    assert state.escalation_pending is not None
    assert state.escalation_pending["reason"] == EscalationReason.SENSITIVE_DOMAIN.value
    assert state.escalation_pending["priority"] == "urgent"


async def test_user_prompt_hook_normal_message():
    state = HookState()
    hook = build_user_prompt_hook(state)
    await hook(
        input_data={"prompt": "Where is my package?"},
        tool_use_id=None,
        context=None,
    )
    assert state.escalation_pending is None


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

async def main():
    runner = TestRunner()

    print("Errors:")
    await runner.run("error_serialization", test_error_serialization())
    await runner.run("error_hierarchy", test_error_hierarchy())

    print("\nServices:")
    await runner.run("crm_lookup_known", test_crm_lookup_known_customer())
    await runner.run("crm_lookup_unknown_raises", test_crm_lookup_unknown_raises())
    await runner.run("crm_find_by_email", test_crm_find_by_email())
    await runner.run("order_lookup", test_order_lookup())
    await runner.run("order_list_by_customer", test_order_list_by_customer())

    print("\nPolicy:")
    await runner.run("refund_ceiling_per_tier", test_refund_ceiling_per_tier())

    print("\nPII redaction:")
    await runner.run("redact_credit_card", test_redact_credit_card())
    await runner.run("redact_ssn", test_redact_ssn())
    await runner.run("redact_passes_clean", test_redact_passes_clean_text())

    print("\nPreToolUse hook:")
    await runner.run("pre_blocks_oversized_refund", test_pre_hook_blocks_oversized_refund())
    await runner.run("pre_allows_within_ceiling", test_pre_hook_allows_within_ceiling())
    await runner.run("pre_blocks_unverified_refund", test_pre_hook_blocks_refund_without_verification())
    await runner.run("pre_redacts_pii", test_pre_hook_redacts_pii_in_input())

    print("\nPostToolUse hook:")
    await runner.run("post_caches_customer", test_post_hook_caches_verified_customer())
    await runner.run("post_resets_failure_count", test_post_hook_resets_failure_count_on_success())

    print("\nFailure hook:")
    await runner.run("failure_triggers_escalation", test_failure_hook_triggers_escalation_after_threshold())
    await runner.run("failure_below_threshold", test_failure_hook_no_escalation_below_threshold())

    print("\nUserPromptSubmit hook:")
    await runner.run("user_prompt_explicit_escalation", test_user_prompt_hook_explicit_escalation())
    await runner.run("user_prompt_sensitive_topic", test_user_prompt_hook_sensitive_topic())
    await runner.run("user_prompt_normal_message", test_user_prompt_hook_normal_message())

    return runner.summary()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
