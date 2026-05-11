"""
Test harness for the support agent.

Exercises the deterministic components — tools, hooks, errors, policy — using
fakes and direct calls. The agent's LLM-driven loop is tested separately with
a live API key (see `demo.py`).

Run:  python -m unittest test_components
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from typing import cast

from claude_agent_sdk.types import HookContext, HookInput

from core.models import (
    CustomerNotFoundError,
    EscalationReason,
    OrderNotFoundError,
    RefundLimitExceededError,
    SupportAgentError,
)
from core.policy import DEFAULT_POLICY
from core.services import CRMService, OrderService
from hooks.compliance import (
    HookState,
    build_post_tool_failure_hook,
    build_post_tool_use_hook,
    build_pre_tool_use_hook,
    build_user_prompt_hook,
    redact_pii,
)
from tools.mcp_tools import (
    TOOL_ESCALATE,
    TOOL_ISSUE_REFUND,
    TOOL_LOOKUP_CUSTOMER,
)


HOOK_BASE_INPUT = {
    "session_id": "session-123",
    "transcript_path": "/path/to/transcript.txt",
    "cwd": "/agent/cwd",
}
HOOK_CONTEXT: HookContext = {"signal": None}


def hook_input(payload: dict[str, object]) -> HookInput:
    return cast(HookInput, payload)


# ---- Errors -------------------------------------------------------------- #

class TestErrors(unittest.IsolatedAsyncioTestCase):
    async def test_error_serialization(self):
        err = RefundLimitExceededError(
            "amount too high",
            context={"amount_usd": 500.0, "ceiling_usd": 50.0},
        )
        payload = err.to_tool_error()
        self.assertEqual(payload["code"], "refund_limit_exceeded")
        self.assertTrue(payload["escalate"])
        self.assertFalse(payload["retryable"])
        self.assertEqual(payload["context"]["amount_usd"], 500.0)

    async def test_error_hierarchy(self):
        for cls in [CustomerNotFoundError, OrderNotFoundError, RefundLimitExceededError]:
            self.assertTrue(issubclass(cls, SupportAgentError), f"{cls} not in hierarchy")


# ---- Services ------------------------------------------------------------ #

class TestServices(unittest.IsolatedAsyncioTestCase):
    async def test_crm_lookup_known_customer(self):
        crm = CRMService()
        customer = await crm.get_customer("C-1001")
        self.assertEqual(customer.tier, "enterprise")
        self.assertEqual(customer.email, "alice@example.com")

    async def test_crm_lookup_unknown_raises(self):
        crm = CRMService()
        with self.assertRaises(CustomerNotFoundError) as ctx:
            await crm.get_customer("C-DOES-NOT-EXIST")
        self.assertEqual(ctx.exception.context["customer_id"], "C-DOES-NOT-EXIST")

    async def test_crm_find_by_email(self):
        crm = CRMService()
        customer = await crm.find_by_email("BOB@example.com")  # case-insensitive
        self.assertEqual(customer.customer_id, "C-1002")

    async def test_order_lookup(self):
        orders = OrderService()
        order = await orders.get_order("O-5001")
        self.assertEqual(order.total_usd, 1_249.00)
        self.assertEqual(order.status, "delivered")

    async def test_order_list_by_customer(self):
        orders = OrderService()
        alice_orders = await orders.list_for_customer("C-1001")
        self.assertEqual(len(alice_orders), 2)


# ---- Policy ------------------------------------------------------------- #

class TestPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_refund_ceiling_per_tier(self):
        policy = DEFAULT_POLICY
        self.assertEqual(policy.refund.ceiling_for("standard"), 50.0)
        self.assertEqual(policy.refund.ceiling_for("pro"), 250.0)
        self.assertEqual(policy.refund.ceiling_for("enterprise"), 1_000.0)
        self.assertEqual(policy.refund.ceiling_for("vip-unknown"), 50.0)


# ---- PII redaction ------------------------------------------------------ #

class TestPIIRedaction(unittest.IsolatedAsyncioTestCase):
    async def test_redact_credit_card(self):
        text = "My card is 4111 1111 1111 1111 please charge it"
        redacted, labels = redact_pii(text)
        self.assertNotIn("4111", redacted)
        self.assertIn("[REDACTED:CC]", redacted)
        self.assertIn("credit_card", labels)

    async def test_redact_ssn(self):
        text = "SSN is 123-45-6789"
        redacted, labels = redact_pii(text)
        self.assertNotIn("123-45-6789", redacted)
        self.assertIn("ssn", labels)

    async def test_redact_passes_clean_text(self):
        text = "Hi, my order arrived broken"
        redacted, labels = redact_pii(text)
        self.assertEqual(redacted, text)
        self.assertEqual(labels, [])


# ---- PreToolUse hook --------------------------------------------------- #

class TestPreToolUseHook(unittest.IsolatedAsyncioTestCase):
    async def test_blocks_oversized_refund(self):
        state = HookState()
        state.verified_customer = {"customer_id": "C-1002", "tier": "standard"}
        hook = build_pre_tool_use_hook(state)

        result = await hook(
            hook_input({
                "hook_event_name": "PreToolUse",
                "tool_name": TOOL_ISSUE_REFUND,
                "tool_input": {"amount_usd": 500.0, "order_id": "O-5003", "customer_id": "C-1002", "reason": "test"},
                "tool_use_id": "tool-1",
                **HOOK_BASE_INPUT,
            }),
            "tool-1",
            HOOK_CONTEXT,
        )
        output = result.get("hookSpecificOutput", {})
        self.assertEqual(output.get("permissionDecision"), "deny")
        self.assertIn("ceiling", output.get("permissionDecisionReason", "").lower())
        assert state.escalation_pending is not None
        self.assertEqual(state.escalation_pending["reason"], EscalationReason.REFUND_THRESHOLD_EXCEEDED.value)  # pylint: disable=unsubscriptable-object

    async def test_allows_within_ceiling(self):
        state = HookState()
        state.verified_customer = {"customer_id": "C-1001", "tier": "enterprise"}
        hook = build_pre_tool_use_hook(state)

        result = await hook(
            hook_input({
                "hook_event_name": "PreToolUse",
                "tool_name": TOOL_ISSUE_REFUND,
                "tool_input": {"amount_usd": 800.0, "order_id": "O-5001", "customer_id": "C-1001", "reason": "test"},
                "tool_use_id": "tool-1",
                **HOOK_BASE_INPUT,
            }),
            "tool-1",
            HOOK_CONTEXT,
        )
        self.assertNotEqual(result.get("hookSpecificOutput", {}).get("permissionDecision"), "deny")
        self.assertIsNone(state.escalation_pending)

    async def test_blocks_refund_without_verification(self):
        state = HookState()
        hook = build_pre_tool_use_hook(state)

        result = await hook(
            hook_input({
                "hook_event_name": "PreToolUse",
                "tool_name": TOOL_ISSUE_REFUND,
                "tool_input": {"amount_usd": 20.0, "order_id": "O-5003", "customer_id": "C-1002", "reason": "test"},
                "tool_use_id": "tool-1",
                **HOOK_BASE_INPUT,
            }),
            "tool-1",
            HOOK_CONTEXT,
        )
        output = result.get("hookSpecificOutput", {})
        self.assertEqual(output.get("permissionDecision"), "deny")
        self.assertIn("verified", output.get("permissionDecisionReason", "").lower())

    async def test_redacts_pii_in_input(self):
        state = HookState()
        hook = build_pre_tool_use_hook(state)
        result = await hook(
            hook_input({
                "hook_event_name": "PreToolUse",
                "tool_name": TOOL_ESCALATE,
                "tool_input": {
                    "ticket_id": "T-1",
                    "reason": "user_requested",
                    "summary": "Customer's card 4111 1111 1111 1111 was charged twice",
                    "priority": "high",
                },
                "tool_use_id": "tool-1",
                **HOOK_BASE_INPUT,
            }),
            "tool-1",
            HOOK_CONTEXT,
        )
        output = result.get("hookSpecificOutput", {})
        self.assertEqual(output.get("permissionDecision"), "allow")
        modified = output.get("updatedInput", {})
        self.assertNotIn("4111", modified.get("summary", ""))
        self.assertIn("[REDACTED:CC]", modified.get("summary", ""))


# ---- PostToolUse hook -------------------------------------------------- #

class TestPostToolUseHook(unittest.IsolatedAsyncioTestCase):
    async def test_caches_verified_customer(self):
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
            hook_input({
                "hook_event_name": "PostToolUse",
                "tool_name": TOOL_LOOKUP_CUSTOMER,
                "tool_input": {"customer_id": "C-1003"},
                "tool_response": fake_result,
                "tool_use_id": "t-1",
                **HOOK_BASE_INPUT,
            }),
            "t-1",
            HOOK_CONTEXT,
        )
        assert state.verified_customer is not None
        assert state.verified_customer["tier"] == "pro"  # pylint: disable=unsubscriptable-object

    async def test_resets_failure_count_on_success(self):
        state = HookState()
        state.failure_counts[TOOL_LOOKUP_CUSTOMER] = 2
        hook = build_post_tool_use_hook(state)
        await hook(
            hook_input({
                "hook_event_name": "PostToolUse",
                "tool_name": TOOL_LOOKUP_CUSTOMER,
                "tool_input": {"customer_id": "C-1001"},
                "tool_response": {"content": [{"type": "text", "text": "{}"}], "isError": False},
                "tool_use_id": "t-1",
                **HOOK_BASE_INPUT,
            }),
            "t-1",
            HOOK_CONTEXT,
        )
        assert state.failure_counts[TOOL_LOOKUP_CUSTOMER] == 0


# ---- Failure hook ------------------------------------------------------ #

class TestPostToolFailureHook(unittest.IsolatedAsyncioTestCase):
    async def test_triggers_escalation_after_threshold(self):
        state = HookState()
        hook = build_post_tool_failure_hook(state)
        for i in range(3):
            await hook(
                hook_input({
                    "hook_event_name": "PostToolUseFailure",
                    "tool_name": TOOL_LOOKUP_CUSTOMER,
                    "tool_input": {"customer_id": "C-1003"},
                    "tool_use_id": f"t-{i}",
                    "error": "timeout",
                    **HOOK_BASE_INPUT,
                }),
                f"t-{i}",
                HOOK_CONTEXT,
            )
        assert state.escalation_pending is not None
        assert state.escalation_pending["reason"] == EscalationReason.REPEATED_FAILURE.value  # pylint: disable=unsubscriptable-object

    async def test_no_escalation_below_threshold(self):
        state = HookState()
        hook = build_post_tool_failure_hook(state)
        for i in range(2):
            await hook(
                hook_input({
                    "hook_event_name": "PostToolUseFailure",
                    "tool_name": TOOL_LOOKUP_CUSTOMER,
                    "tool_input": {"customer_id": "C-1003"},
                    "tool_use_id": f"t-{i}",
                    "error": "timeout",
                    **HOOK_BASE_INPUT,
                }),
                f"t-{i}",
                HOOK_CONTEXT,
            )
        assert state.escalation_pending is None


# ---- User prompt hook -------------------------------------------------- #

class TestUserPromptHook(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_escalation(self):
        state = HookState()
        hook = build_user_prompt_hook(state)
        await hook(
            hook_input({
                "hook_event_name": "UserPromptSubmit",
                "prompt": "I want to speak to a human right now",
                **HOOK_BASE_INPUT,
            }),
            None,
            HOOK_CONTEXT,
        )
        assert state.escalation_pending is not None
        assert state.escalation_pending["reason"] == EscalationReason.EXPLICIT_USER_REQUEST.value  # pylint: disable=unsubscriptable-object

    async def test_sensitive_topic(self):
        state = HookState()
        hook = build_user_prompt_hook(state)
        await hook(
            hook_input({
                "hook_event_name": "UserPromptSubmit",
                "prompt": "I'm considering legal action over this",
                **HOOK_BASE_INPUT,
            }),
            None,
            HOOK_CONTEXT,
        )
        assert state.escalation_pending is not None
        assert state.escalation_pending["reason"] == EscalationReason.SENSITIVE_DOMAIN.value  # pylint: disable=unsubscriptable-object
        assert state.escalation_pending["priority"] == "urgent"  # pylint: disable=unsubscriptable-object

    async def test_normal_message(self):
        state = HookState()
        hook = build_user_prompt_hook(state)
        await hook(
            hook_input({
                "hook_event_name": "UserPromptSubmit",
                "prompt": "Where is my package?",
                **HOOK_BASE_INPUT,
            }),
            None,
            HOOK_CONTEXT,
        )
        assert state.escalation_pending is None


if __name__ == "__main__":
    unittest.main()
