"""
Mock backend services. In production these are HTTP clients to your CRM,
order system, and ticketing platform. The interface is intentionally narrow
so swapping in real services is a drop-in change.

Each method raises a typed error from `core.models` rather than returning
None or a tuple — this matters because the agent's failure hook dispatches
on exception type.
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

from .models import (
    Customer,
    CustomerNotFoundError,
    ExternalServiceError,
    Order,
    OrderNotFoundError,
)


class CRMService:
    """Customer lookup. In prod: GET /v1/customers/{id} on your CRM."""

    def __init__(self, *, failure_rate: float = 0.0):
        self._customers: dict[str, Customer] = {
            "C-1001": Customer(
                customer_id="C-1001",
                email="alice@example.com",
                name="Alice Chen",
                tier="enterprise",
                lifetime_value_usd=48_000.0,
            ),
            "C-1002": Customer(
                customer_id="C-1002",
                email="bob@example.com",
                name="Bob Martinez",
                tier="standard",
                lifetime_value_usd=320.0,
            ),
            "C-1003": Customer(
                customer_id="C-1003",
                email="carol@example.com",
                name="Carol Singh",
                tier="pro",
                lifetime_value_usd=4_200.0,
            ),
        }
        self._failure_rate = failure_rate

    async def get_customer(self, customer_id: str) -> Customer:
        await asyncio.sleep(0.02)
        if random.random() < self._failure_rate:
            raise ExternalServiceError(
                "CRM lookup timed out",
                context={"service": "crm", "customer_id": customer_id},
            )
        if customer_id not in self._customers:
            raise CustomerNotFoundError(
                f"No customer found with id {customer_id}",
                context={"customer_id": customer_id},
            )
        return self._customers[customer_id]

    async def find_by_email(self, email: str) -> Customer:
        await asyncio.sleep(0.02)
        for c in self._customers.values():
            if c.email.lower() == email.lower():
                return c
        raise CustomerNotFoundError(
            f"No customer with email {email}",
            context={"email": email},
        )


class OrderService:
    """Order lookup."""

    def __init__(self):
        now = datetime.now(timezone.utc)
        self._orders: dict[str, Order] = {
            "O-5001": Order("O-5001", "C-1001", 1_249.00, "delivered", now - timedelta(days=14)),
            "O-5002": Order("O-5002", "C-1001", 89.99, "shipped", now - timedelta(days=2)),
            "O-5003": Order("O-5003", "C-1002", 42.50, "delivered", now - timedelta(days=30)),
            "O-5004": Order("O-5004", "C-1003", 320.00, "pending", now - timedelta(hours=6)),
        }

    async def get_order(self, order_id: str) -> Order:
        await asyncio.sleep(0.02)
        if order_id not in self._orders:
            raise OrderNotFoundError(
                f"No order found with id {order_id}",
                context={"order_id": order_id},
            )
        return self._orders[order_id]

    async def list_for_customer(self, customer_id: str) -> list[Order]:
        await asyncio.sleep(0.02)
        return [o for o in self._orders.values() if o.customer_id == customer_id]


class RefundService:
    """Issues refunds. In prod: POST /v1/refunds on your payments service."""

    def __init__(self):
        self.issued: list[dict] = []   # audit log

    async def issue_refund(
        self,
        *,
        order_id: str,
        amount_usd: float,
        reason: str,
        approved_by: str,
    ) -> str:
        await asyncio.sleep(0.05)
        refund_id = f"R-{len(self.issued) + 1:04d}"
        self.issued.append({
            "refund_id": refund_id,
            "order_id": order_id,
            "amount_usd": amount_usd,
            "reason": reason,
            "approved_by": approved_by,
            "issued_at": datetime.now(timezone.utc).isoformat(),
        })
        return refund_id


class TicketingService:
    """Creates and updates support tickets."""

    def __init__(self):
        self.tickets: dict[str, dict] = {}

    async def create_ticket(self, payload: dict) -> str:
        await asyncio.sleep(0.02)
        ticket_id = payload.get("ticket_id") or f"T-{len(self.tickets) + 1:04d}"
        self.tickets[ticket_id] = {**payload, "ticket_id": ticket_id}
        return ticket_id

    async def update_ticket(self, ticket_id: str, updates: dict) -> None:
        await asyncio.sleep(0.02)
        if ticket_id not in self.tickets:
            self.tickets[ticket_id] = {"ticket_id": ticket_id}
        self.tickets[ticket_id].update(updates)


# --------------------------------------------------------------------------- #
# Service registry — passed to tools so they can be unit-tested with fakes    #
# --------------------------------------------------------------------------- #

class ServiceRegistry:
    def __init__(
        self,
        crm: CRMService | None = None,
        orders: OrderService | None = None,
        refunds: RefundService | None = None,
        tickets: TicketingService | None = None,
    ):
        self.crm = crm or CRMService()
        self.orders = orders or OrderService()
        self.refunds = refunds or RefundService()
        self.tickets = tickets or TicketingService()
