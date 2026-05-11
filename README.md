# Customer Support Agent

An AI-powered customer support agent built on the Claude Agent SDK. Handles inquiries, resolves issues within policy, and escalates complex cases to human agents using a deterministic compliance layer.

## What it does

A customer sends a message. The agent:

1. Verifies identity by looking up the customer
2. Searches the knowledge base for self-service answers
3. Looks up orders and issues refunds within its auto-approval ceiling
4. Escalates to a human when policy, sentiment, or topic requires it
5. Returns a structured result with the final answer, ticket ID, and a full audit trail

## Architecture

```
support_agent/
├── agent.py                  Orchestrator: wires SDK + tools + hooks
├── demo.py                   Five end-to-end scenarios
├── core/
│   ├── models.py             Domain types + structured error hierarchy
│   ├── policy.py             Refund ceilings, escalation triggers (data, not code)
│   └── services.py           CRM / Orders / Refunds / Ticketing (mock backends)
├── tools/
│   └── mcp_tools.py          In-process MCP server with 6 tools
├── hooks/
│   └── compliance.py         PreToolUse, PostToolUse, Failure, UserPromptSubmit
└── tests/
    └── test_components.py    22 deterministic tests (no API key needed)
```

## How each requirement maps to code

### Agent SDK implementation

`agent.py` constructs `ClaudeAgentOptions` with `model`, `system_prompt`, `mcp_servers`, `allowed_tools`, and `hooks`, then drives the loop with `ClaudeSDKClient` and consumes streaming `AssistantMessage`s. A fresh MCP server is built per session so the tools close over per-session service state — this is what makes the agent safe to run concurrently.

### MCP tools

Six tools in `tools/mcp_tools.py`, registered via `@tool` and `create_sdk_mcp_server`. They run in-process (no IPC, no separate server). Each tool catches `SupportAgentError` and returns it as structured tool content (with `isError: True`) so Claude sees the failure and can adapt — never as a raw exception.

| Tool | Purpose |
|---|---|
| `lookup_customer` | Identity verification (must be called before any state-changing action) |
| `get_order` / `list_customer_orders` | Order context |
| `search_knowledge_base` | Self-service answers — agent must try this before escalating policy questions |
| `issue_refund` | Auto-approves within tier ceiling, errors above it |
| `escalate_to_human` | Hand-off with structured reason and summary |

### Escalation pattern design

Escalation can be triggered from four independent sources, each handled in its own layer:

| Trigger | Where it fires | Reason set |
|---|---|---|
| User explicitly asks for a human | `UserPromptSubmit` hook (before Claude sees the message) | `EXPLICIT_USER_REQUEST` |
| Sensitive topic (legal, medical, data deletion) | `UserPromptSubmit` hook | `SENSITIVE_DOMAIN` |
| Refund exceeds tier ceiling | `PreToolUse` hook denies the call | `REFUND_THRESHOLD_EXCEEDED` |
| Same tool fails N times | `PostToolUseFailure` hook | `REPEATED_FAILURE` |

The model is also instructed to call `escalate_to_human` for low-confidence answers and to provide a handoff summary so a human can pick up without re-asking the customer.

`EscalationReason` is a closed enum — every reason maps to a different human workflow downstream. Don't add a new reason without adding a runbook entry.

### Hook-based compliance enforcement

The hooks in `hooks/compliance.py` are the deterministic control layer. The LLM is free to be clever in its prompting, but these functions enforce non-negotiable rules:

- **PreToolUse** — Blocks refunds above the tier ceiling, blocks state-changing actions before identity is verified, and redacts PII (credit cards, SSNs, API keys) from tool inputs before they reach downstream services.
- **PostToolUse** — Caches the verified customer (so subsequent PreToolUse calls can authorize state changes), resets failure counters on success, writes audit log entries.
- **PostToolUseFailure** — Counts consecutive failures per tool and triggers escalation when the threshold is hit.
- **UserPromptSubmit** — Scans the user message for explicit handoff requests and sensitive topics before the LLM sees it.

Hooks share a per-session `HookState` object — never module-level globals — so concurrent agent runs for different customers stay isolated.

### Structured error handling

`SupportAgentError` is the base class, with a stable `code`, `escalate` flag, and `retryable` flag. Subclasses (`CustomerNotFoundError`, `OrderNotFoundError`, `AuthenticationError`, `PolicyViolationError`, `RefundLimitExceededError`, `ExternalServiceError`) cover specific failure categories. Tools catch the base class and return `error.to_tool_error()` — the dispatch on type happens in the failure hook, not by parsing strings.

Two layers enforce the refund ceiling: the `PreToolUse` hook blocks the call before it runs, and `issue_refund` itself re-checks the ceiling and raises `RefundLimitExceededError` if called directly. Defense in depth — tools never trust upstream.

## Running

```bash
pip install claude-agent-sdk

# Deterministic tests (no API key needed) — 22 tests
python -m support_agent.tests.test_components

# Live demo — runs five scenarios end-to-end
export ANTHROPIC_API_KEY=...
python -m support_agent.demo
```

## Design notes

**Why in-process MCP?** The SDK supports two kinds of MCP servers: subprocess-based (for third-party tools) and in-process via `create_sdk_mcp_server`. In-process tools share the agent's address space, which means no IPC overhead and direct access to the service registry — important for performance when every customer message kicks off several tool calls.

**Why hooks instead of trusting the system prompt?** The system prompt tells the model *not* to issue oversized refunds, but a long conversation, a confused customer, or a clever prompt-injection attempt could cause it to try anyway. Hooks make that impossible — the call is denied at the SDK boundary regardless of what the model decided. Prompts are guidance, hooks are enforcement.

**Why a per-session HookState?** A server hosting this agent will run many sessions concurrently. Module-level state would mean one customer's verified-identity status could be read by another customer's session. The state object is constructed fresh in `handle_inquiry` and passed to each hook factory, so each session is fully isolated.
