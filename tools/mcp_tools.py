"""
MCP tools exposed to the agent.

Built as an in-process SDK MCP server (via `create_sdk_mcp_server`) so the
tools share the same Python process as the agent — no IPC, no separate
server to manage, full type safety.

Each tool follows the same pattern:
  1. Validate input.
  2. Call the underlying service.
  3. Catch typed `SupportAgentError`s and return them as structured tool
     errors (so Claude can see them and adapt). Never let raw stack traces
     leak into the tool result.
"""
from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from ..core.models import (
    EscalationReason,
    RefundLimitExceededError,
    SupportAgentError,
)
from ..core.policy import DEFAULT_POLICY
from ..core.services import ServiceRegistry


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a successful tool result in MCP's expected content shape."""
    import json
    return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}


def _err(error: SupportAgentError) -> dict[str, Any]:
    """Wrap a typed error so Claude sees it as a tool result, not a crash."""
    import json
    return {
        "content": [{"type": "text", "text": json.dumps(error.to_tool_error())}],
        "isError": True,
    }


# --------------------------------------------------------------------------- #
# Tool factory                                                                #
# --------------------------------------------------------------------------- #

def build_support_mcp_server(services: ServiceRegistry, policy=DEFAULT_POLICY):
    """Construct the in-process MCP server with all support tools bound to
    the given service registry. Returning a fresh server per agent run keeps
    tests isolated — pass a registry built from fakes.
    """

    # --- Customer lookup -------------------------------------------------- #
    @tool(
        "lookup_customer",
        "Look up a customer by their customer_id (e.g. 'C-1001') OR by email. "
        "Returns the customer's name, tier, and lifetime value. Use this "
        "FIRST in any conversation to verify identity before taking actions.",
        {"customer_id": str, "email": str},
    )
    async def lookup_customer(args: dict[str, Any]) -> dict[str, Any]:
        try:
            cid = (args.get("customer_id") or "").strip()
            email = (args.get("email") or "").strip()
            if cid:
                customer = await services.crm.get_customer(cid)
            elif email:
                customer = await services.crm.find_by_email(email)
            else:
                raise SupportAgentError("Provide either customer_id or email")
            return _ok({
                "customer_id": customer.customer_id,
                "name": customer.name,
                "email": customer.email,
                "tier": customer.tier,
                "lifetime_value_usd": customer.lifetime_value_usd,
            })
        except SupportAgentError as e:
            return _err(e)

    # --- Order lookup ----------------------------------------------------- #
    @tool(
        "get_order",
        "Fetch details for a single order by order_id (e.g. 'O-5001'). "
        "Returns total, status, and creation date.",
        {"order_id": str},
    )
    async def get_order(args: dict[str, Any]) -> dict[str, Any]:
        try:
            order = await services.orders.get_order(args["order_id"])
            return _ok({
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "total_usd": order.total_usd,
                "status": order.status,
                "created_at": order.created_at.isoformat(),
            })
        except SupportAgentError as e:
            return _err(e)

    @tool(
        "list_customer_orders",
        "List all orders for a customer. Use after lookup_customer when the "
        "user references 'my recent order' without giving an order_id.",
        {"customer_id": str},
    )
    async def list_customer_orders(args: dict[str, Any]) -> dict[str, Any]:
        try:
            orders = await services.orders.list_for_customer(args["customer_id"])
            return _ok({
                "orders": [
                    {
                        "order_id": o.order_id,
                        "total_usd": o.total_usd,
                        "status": o.status,
                        "created_at": o.created_at.isoformat(),
                    }
                    for o in orders
                ]
            })
        except SupportAgentError as e:
            return _err(e)

    # --- Refunds ---------------------------------------------------------- #
    @tool(
        "issue_refund",
        "Issue a refund for an order. The customer's tier determines the "
        "auto-approval ceiling — refunds above that threshold will be "
        "blocked by policy and must be escalated to a human. "
        "Always lookup_customer and get_order first to validate.",
        {
            "order_id": str,
            "customer_id": str,
            "amount_usd": float,
            "reason": str,
        },
    )
    async def issue_refund(args: dict[str, Any]) -> dict[str, Any]:
        try:
            customer = await services.crm.get_customer(args["customer_id"])
            amount = float(args["amount_usd"])
            ceiling = policy.refund.ceiling_for(customer.tier)

            # Belt-and-suspenders: the PreToolUse hook also checks this, but
            # tools never trust upstream — defense in depth.
            if amount > ceiling or amount > policy.refund.hard_cap_usd:
                raise RefundLimitExceededError(
                    f"Refund of ${amount:.2f} exceeds auto-approval ceiling "
                    f"of ${ceiling:.2f} for {customer.tier} tier",
                    context={
                        "amount_usd": amount,
                        "ceiling_usd": ceiling,
                        "tier": customer.tier,
                    },
                )

            refund_id = await services.refunds.issue_refund(
                order_id=args["order_id"],
                amount_usd=amount,
                reason=args["reason"],
                approved_by="agent",
            )
            return _ok({
                "refund_id": refund_id,
                "amount_usd": amount,
                "status": "issued",
            })
        except SupportAgentError as e:
            return _err(e)

    # --- Knowledge base --------------------------------------------------- #
    KB = {
        "shipping": "Standard shipping is 3–5 business days. Express is 1–2.",
        "returns": "Items can be returned within 30 days of delivery for a "
                   "full refund. Items must be unused and in original packaging.",
        "warranty": "All products carry a 1-year manufacturer warranty. "
                    "Extended coverage available at checkout.",
        "account": "Account changes (email, password) must be made at "
                   "account.example.com — agents cannot make them on your behalf.",
    }

    @tool(
        "search_knowledge_base",
        "Search the support knowledge base for self-service answers about "
        "shipping, returns, warranty, or account topics. Use this BEFORE "
        "escalating any general policy question — most are answered here.",
        {"query": str},
    )
    async def search_knowledge_base(args: dict[str, Any]) -> dict[str, Any]:
        q = args["query"].lower()
        hits = [{"topic": k, "content": v} for k, v in KB.items() if k in q or any(w in v.lower() for w in q.split())]
        return _ok({"results": hits, "query": args["query"]})

    # --- Escalation ------------------------------------------------------- #
    @tool(
        "escalate_to_human",
        "Hand the conversation off to a human agent. ALWAYS use this when: "
        "(a) the customer explicitly asks for a human, "
        "(b) the issue involves legal / safety / data-deletion topics, "
        "(c) you've tried the knowledge base and tools and still cannot "
        "resolve, or (d) a refund or action requires approval beyond your "
        "ceiling. Provide a clear `summary` of what was attempted so the "
        "human can pick up without re-asking the customer.",
        {
            "ticket_id": str,
            "reason": str,         # one of EscalationReason values
            "summary": str,        # human-readable handoff notes
            "priority": str,       # "low" | "normal" | "high" | "urgent"
        },
    )
    async def escalate_to_human(args: dict[str, Any]) -> dict[str, Any]:
        try:
            reason_raw = args.get("reason", "")
            try:
                reason = EscalationReason(reason_raw)
            except ValueError:
                reason = EscalationReason.LOW_CONFIDENCE  # default fallback

            await services.tickets.update_ticket(
                args["ticket_id"],
                {
                    "status": "escalated",
                    "escalation_reason": reason.value,
                    "escalation_summary": args["summary"],
                    "priority": args.get("priority", "normal"),
                },
            )
            return _ok({
                "ticket_id": args["ticket_id"],
                "escalated": True,
                "reason": reason.value,
                "message": "Handed off to human agent. They will see the full conversation history.",
            })
        except SupportAgentError as e:
            return _err(e)

    return create_sdk_mcp_server(
        name="support-tools",
        version="1.0.0",
        tools=[
            lookup_customer,
            get_order,
            list_customer_orders,
            issue_refund,
            search_knowledge_base,
            escalate_to_human,
        ],
    )


# Tool names as Claude sees them (mcp__<server>__<tool>). Used by the agent's
# allowed_tools list and by hook matchers. Centralized to avoid typos.
TOOL_LOOKUP_CUSTOMER = "mcp__support-tools__lookup_customer"
TOOL_GET_ORDER = "mcp__support-tools__get_order"
TOOL_LIST_ORDERS = "mcp__support-tools__list_customer_orders"
TOOL_ISSUE_REFUND = "mcp__support-tools__issue_refund"
TOOL_SEARCH_KB = "mcp__support-tools__search_knowledge_base"
TOOL_ESCALATE = "mcp__support-tools__escalate_to_human"

ALL_TOOLS = [
    TOOL_LOOKUP_CUSTOMER,
    TOOL_GET_ORDER,
    TOOL_LIST_ORDERS,
    TOOL_ISSUE_REFUND,
    TOOL_SEARCH_KB,
    TOOL_ESCALATE,
]
