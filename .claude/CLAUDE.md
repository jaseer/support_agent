# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a customer support agent built on the Claude Agent SDK. It verifies customer identity, searches a knowledge base, issues refunds within policy limits, and escalates complex cases to humans using a deterministic compliance layer. The agent combines LLM flexibility with hardware-enforced guardrails.

## Architecture & Key Design Patterns

### Component Layout

```
support_agent/
├── agent.py                  # Orchestrator: wires SDK + tools + hooks + services
├── demo.py                   # Five end-to-end scenarios requiring ANTHROPIC_API_KEY
├── core/
│   ├── models.py             # Domain types, error hierarchy, enums (EscalationReason)
│   ├── policy.py             # Refund ceilings by tier (data-driven, not code)
│   └── services.py           # Mock backends: CRM, Orders, Refunds, Ticketing
├── tools/
│   └── mcp_tools.py          # Six tools in-process MCP server
├── hooks/
│   └── compliance.py         # Four hook layers: PreToolUse, PostToolUse, PostToolUseFailure, UserPromptSubmit
└── tests/
    └── test_components.py    # 22 deterministic tests (no API key needed)
```

### Core Concepts

**In-Process MCP Tools**: Tools run in the agent's Python process via `create_sdk_mcp_server`, not as subprocesses. This eliminates IPC overhead and gives tools direct access to the service registry — critical for multi-tool sequences on each message.

**Hook-Based Policy Enforcement**: The system prompt *tells* the model not to issue oversized refunds; hooks *prevent* it. Compliance lives in the four hook factories in `hooks/compliance.py`, not in prompting alone:
- `PreToolUse` — blocks calls above refund ceiling, blocks state changes before identity verification, redacts PII
- `PostToolUse` — caches verified customer, resets failure counters, writes audit log
- `PostToolUseFailure` — counts consecutive tool failures, escalates after N retries
- `UserPromptSubmit` — scans raw user input (before LLM sees it) for explicit hand-off requests and sensitive topics

**Per-Session State Isolation**: Each agent session creates a fresh `HookState` object shared by all hooks for that session. There are no module-level globals. This makes the agent safe to run concurrently for multiple customers.

**Typed Error Hierarchy**: `SupportAgentError` is the base; subclasses cover specific failure modes (`CustomerNotFoundError`, `RefundLimitExceededError`, etc.). Tools catch the base class and return errors as structured tool content (with `isError: True`), never raw exceptions. Hooks dispatch on the specific error type.

**Escalation Enum**: `EscalationReason` is a closed enum with no string-based reasons. Every value maps 1:1 to a human runbook. Do not add a new reason without updating the downstream runbook.

### Key Invariants

- **Identity verification must happen before refunds**: The `PreToolUse` hook blocks any state-changing action (refunds, account changes) unless `lookup_customer` has been called and cached the customer in `HookState.verified_customer`.
- **Refund ceiling is enforced twice**: Once in the `PreToolUse` hook (blocking the call) and again in `issue_refund` itself (defense in depth).
- **Tools never raise exceptions to Claude**: They catch `SupportAgentError` and return it as structured content. Only fatal exceptions (bugs, not expected failures) should escape.

## Common Development Tasks

### Run Tests

```bash
python -m unittest discover -s tests
```

Tests are deterministic and require no API key. They exercise the tools, hooks, error hierarchy, and policy logic using fakes and direct calls.

### Run a Specific Test

```bash
python -m unittest tests.test_components.TestPolicyEnforcement.test_refund_above_ceiling
```

### Run the Live Demo

Requires `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_API_KEY=sk-...
python -m support_agent.demo
```

This runs five end-to-end scenarios:
1. Knowledge base hit (no escalation)
2. Small refund within ceiling
3. Oversized refund (hook blocks it, forces escalation)
4. Explicit human request (UserPromptSubmit hook escalation)
5. Sensitive topic (pre-escalation before LLM sees it)

### Adding a New Tool

1. Implement the tool function in `tools/mcp_tools.py`, decorated with `@tool`.
2. Have it call the underlying service from `ServiceRegistry`.
3. Catch `SupportAgentError` and return `_err(error)`.
4. Register the tool name in `ALL_TOOLS`.
5. Update the system prompt in `agent.py` to teach the model when to use it.
6. If the tool can change state or is sensitive, add hook logic in `hooks/compliance.py`.
7. Test the new tool's happy path and error cases in `tests/test_components.py`.

### Adding a New Escalation Reason

1. Add the reason to the `EscalationReason` enum in `core/models.py`.
2. Add a hook in `hooks/compliance.py` that detects the condition and sets this reason.
3. Document the runbook downstream (in your ticketing system).
4. Add a test case in `tests/test_components.py` that triggers the escalation.

### Debugging a Hook

Hooks receive `HookInput` with `session_id`, `transcript_path`, and `cwd`. The `transcript_path` contains the raw conversation history; read it to understand what the agent has seen and said so far. The `HookState` object in each hook factory tracks `verified_customer`, `failure_count_by_tool`, and audit log entries.

## Important Notes

- **Avoid module-level state**: Use `HookState` objects created fresh per session; never store customer or session data at module scope.
- **System prompt is guidance, hooks are enforcement**: The system prompt should describe the policy to the model. Hooks make it non-negotiable.
- **Keep EscalationReason small**: Every reason should drive a different human workflow. Avoid generic reasons like `NEEDS_ESCALATION`.
- **Always use typed exceptions**: Parse errors by exception type in hooks, not by string matching.
