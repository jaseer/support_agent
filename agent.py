"""
Customer support agent — orchestrator.

Wires the Agent SDK together with our MCP tools and compliance hooks. The
public surface is intentionally tiny: instantiate the agent, call
`handle_inquiry(...)`, and get back a structured response that includes the
final answer, whether escalation occurred, and a full audit trail.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    TextBlock,
)

from .core.models import EscalationReason, Ticket, TicketStatus
from .core.policy import AgentPolicy, DEFAULT_POLICY
from .core.services import ServiceRegistry
from .hooks.compliance import (
    HookState,
    build_post_tool_failure_hook,
    build_post_tool_use_hook,
    build_pre_tool_use_hook,
    build_user_prompt_hook,
)
from .tools.mcp_tools import ALL_TOOLS, build_support_mcp_server

log = logging.getLogger("support_agent")


# --------------------------------------------------------------------------- #
# System prompt                                                               #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are a customer support agent for an e-commerce platform. Your job is to
resolve customer inquiries efficiently while knowing when to escalate.

OPERATING PRINCIPLES

1. VERIFY IDENTITY FIRST. Before taking any state-changing action (refunds,
   account changes), call `lookup_customer` and confirm you have the right
   account. Refer to the customer by name once verified.

2. USE THE KNOWLEDGE BASE. For general policy questions (shipping, returns,
   warranty, account changes), call `search_knowledge_base` BEFORE escalating.
   Most general questions are answered there.

3. RESOLVE WHAT YOU CAN. You can issue refunds within your auto-approval
   ceiling, look up orders, and answer policy questions. Do this work; don't
   punt to a human if you can solve it yourself.

4. ESCALATE WHEN APPROPRIATE. Call `escalate_to_human` when:
   - The customer explicitly asks to speak with a human.
   - You hit a refund or action that exceeds your auto-approval ceiling
     (the system will tell you when this happens).
   - The issue involves legal action, data deletion requests, injury, or
     other sensitive domains.
   - You've genuinely tried and cannot resolve the issue.
   - A tool keeps failing and you can't make progress.

   When escalating, write a `summary` that lets a human agent pick up
   without re-asking the customer. Include: who they are, what they want,
   what you tried, and what the blocker is.

5. BE WARM AND CLEAR. Customers contacting support are often frustrated.
   Acknowledge the problem, explain what you're doing, and confirm next
   steps. Don't pad with apologies or filler.

6. NEVER fabricate refund IDs, order IDs, ticket IDs, or policy details.
   If you don't know, look it up or say so.

7. NEVER ask for credit card numbers, SSNs, passwords, or API keys in chat.
   These are redacted by the platform and shouldn't be needed for support.

When you've fully resolved an inquiry OR escalated it, end your message
with a brief confirmation of what happens next so the customer knows where
they stand.
"""


# --------------------------------------------------------------------------- #
# Result shape                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class AgentResult:
    ticket_id: str
    final_message: str
    escalated: bool
    escalation_reason: str | None = None
    escalation_summary: str | None = None
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    raw_messages: list[Any] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #

class CustomerSupportAgent:
    """One agent per support session. Construct, call `handle_inquiry`, done.

    Concurrency: each session has its own HookState. Don't share a single
    agent instance across users — construct one per conversation.
    """

    def __init__(
        self,
        services: ServiceRegistry | None = None,
        policy: AgentPolicy = DEFAULT_POLICY,
        model: str = "claude-opus-4-7",
    ):
        self.services = services or ServiceRegistry()
        self.policy = policy
        self.model = model

    def _build_options(self, state: HookState, ticket: Ticket) -> ClaudeAgentOptions:
        """Construct the per-session SDK options.

        Note we re-construct the MCP server per session so its tools close
        over a session-specific service registry — this is what makes the
        agent safe to run concurrently for different customers.
        """
        mcp_server = build_support_mcp_server(self.services, self.policy)

        # The system prompt is augmented with this session's ticket id so the
        # model uses it consistently when calling escalate_to_human.
        session_prompt = (
            SYSTEM_PROMPT
            + f"\n\nCURRENT TICKET ID: {ticket.ticket_id}\n"
            + "When calling escalate_to_human, use this ticket_id."
        )

        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=session_prompt,
            mcp_servers={"support-tools": mcp_server},
            allowed_tools=ALL_TOOLS,
            permission_mode="default",
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        # Match all tools — we want every call gated.
                        matcher=None,
                        hooks=[cast(Any, build_pre_tool_use_hook(state, self.policy))],
                    ),
                ],
                "PostToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=[cast(Any, build_post_tool_use_hook(state, self.policy))],
                    ),
                ],
                "PostToolUseFailure": [
                    HookMatcher(
                        matcher=None,
                        hooks=[cast(Any, build_post_tool_failure_hook(state, self.policy))],
                    ),
                ],
                "UserPromptSubmit": [
                    HookMatcher(
                        matcher=None,
                        hooks=[cast(Any, build_user_prompt_hook(state, self.policy))],
                    ),
                ],
            },
        )

    async def handle_inquiry(
        self,
        message: str,
        *,
        customer_hint: str | None = None,
        ticket_id: str | None = None,
    ) -> AgentResult:
        """Handle one customer inquiry end-to-end.

        Args:
            message: the customer's message.
            customer_hint: optional pre-known customer_id or email — included
                in the prompt so the agent doesn't have to ask.
            ticket_id: optional existing ticket; otherwise a new one is created.
        """
        ticket = Ticket(
            ticket_id=ticket_id or f"T-{uuid4().hex[:8]}",
            subject=message[:60],
        )
        await self.services.tickets.create_ticket({
            "ticket_id": ticket.ticket_id,
            "subject": ticket.subject,
            "status": TicketStatus.IN_PROGRESS.value,
        })

        state = HookState()
        options = self._build_options(state, ticket)

        # Compose the user-facing prompt. If we have a customer hint, surface
        # it so the agent can verify identity without a back-and-forth.
        prompt_parts = []
        if customer_hint:
            prompt_parts.append(f"[Customer context: {customer_hint}]")
        prompt_parts.append(message)
        full_prompt = "\n\n".join(prompt_parts)

        raw_messages: list[Any] = []
        final_text_chunks: list[str] = []

        async with ClaudeSDKClient(options=options) as client:
            await client.query(full_prompt)
            async for msg in client.receive_response():
                raw_messages.append(msg)
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            final_text_chunks.append(block.text)

        final_message = "\n".join(t for t in final_text_chunks if t).strip()

        # Determine escalation status from hook state + ticket state.
        escalated = state.escalation_pending is not None
        escalation_reason = None
        escalation_summary = None
        if escalated:
            escalation_reason = state.escalation_pending["reason"]
            escalation_summary = state.escalation_pending["summary"]
            ticket.status = TicketStatus.ESCALATED
            try:
                ticket.escalation_reason = EscalationReason(escalation_reason)
            except ValueError:
                ticket.escalation_reason = EscalationReason.LOW_CONFIDENCE
            ticket.escalation_notes = escalation_summary
            await self.services.tickets.update_ticket(
                ticket.ticket_id,
                {
                    "status": TicketStatus.ESCALATED.value,
                    "escalation_reason": escalation_reason,
                    "escalation_summary": escalation_summary,
                },
            )
        else:
            ticket.status = TicketStatus.RESOLVED
            await self.services.tickets.update_ticket(
                ticket.ticket_id,
                {"status": TicketStatus.RESOLVED.value},
            )

        return AgentResult(
            ticket_id=ticket.ticket_id,
            final_message=final_message,
            escalated=escalated,
            escalation_reason=escalation_reason,
            escalation_summary=escalation_summary,
            audit_log=state.audit_log,
            raw_messages=raw_messages,
        )
