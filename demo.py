"""
End-to-end demo of the support agent. Requires ANTHROPIC_API_KEY in the env.

Five scenarios are run, each exercising a different code path:

  1. Self-service KB hit          — agent resolves without escalation
  2. Order lookup + small refund  — within auto-approval ceiling
  3. Oversized refund             — blocked by hook → forced escalation
  4. Explicit human request       — UserPromptSubmit hook escalation
  5. Sensitive topic              — sensitive-domain escalation

Run:  python -m support_agent.demo
"""
from __future__ import annotations

import asyncio
import os
import sys

from core.services import ServiceRegistry
from agent import CustomerSupportAgent

SCENARIOS = [
    {
        "name": "1. Self-service knowledge base hit",
        "message": "What's your return policy? I bought something two weeks ago.",
        "customer_hint": None,
    },
    {
        "name": "2. Small refund within ceiling (standard tier, $42 order)",
        "message": (
            "My order O-5003 arrived damaged. The box was crushed and the "
            "contents are unusable. I'd like a refund please."
        ),
        "customer_hint": "customer_id=C-1002 (Bob Martinez, standard tier)",
    },
    {
        "name": "3. Refund above ceiling (standard tier, requesting $500)",
        "message": (
            "I want a full refund of $500 for damages caused by your product. "
            "Order O-5003."
        ),
        "customer_hint": "customer_id=C-1002 (Bob Martinez, standard tier)",
    },
    {
        "name": "4. Explicit request for a human",
        "message": "Stop. I want to speak to a human supervisor immediately.",
        "customer_hint": "customer_id=C-1003",
    },
    {
        "name": "5. Sensitive topic (legal action)",
        "message": (
            "Your product injured me and I'm considering legal action. "
            "I need to know my options."
        ),
        "customer_hint": "customer_id=C-1001",
    },
]


async def run_scenario(agent: CustomerSupportAgent, scenario: dict):
    print(f"\n{'=' * 70}")
    print(scenario["name"])
    print("=" * 70)
    print(f"Customer: {scenario['message']}\n")

    result = await agent.handle_inquiry(
        scenario["message"],
        customer_hint=scenario["customer_hint"],
    )

    print(f"Agent: {result.final_message}\n")
    print(f"Ticket: {result.ticket_id}")
    print(f"Escalated: {result.escalated}")
    if result.escalated:
        print(f"  Reason:  {result.escalation_reason}")
        print(f"  Summary: {result.escalation_summary}")
    print(f"Tool calls: {len([e for e in result.audit_log if e['phase'] == 'pre'])}")


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — set it before running the demo.")
        return 1

    services = ServiceRegistry()
    agent = CustomerSupportAgent(services=services)

    for scenario in SCENARIOS:
        try:
            await run_scenario(agent, scenario)
        except Exception as e:
            print(f"\nScenario failed: {e}")

    # Recap of all tickets created in the run
    print(f"\n\n{'=' * 70}")
    print("Ticket summary")
    print("=" * 70)
    for ticket_id, t in services.tickets.tickets.items():
        status = t.get("status", "?")
        reason = t.get("escalation_reason", "")
        print(f"  {ticket_id:12} {status:12} {reason}")

    return 0


if __name__ == "__main__":
    # Run the main function in an asyncio event loop and exit with its return code
    sys.exit(asyncio.run(main()))
