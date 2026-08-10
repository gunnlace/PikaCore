# PikaCore Repository Guidance

## Scope

- PikaCore is a fork of CoreCoder and preserves native function calling.
- Read `docs/PIKACORE_DESIGN.md` before architectural changes.
- Implement only the active design phase; do not pull later phases forward.
- V1 has Working Memory only. Do not add Durable Memory, providers, prompt-cache logic, or an agent framework.

## Contracts

- Keep assistant tool calls paired with exactly one tool result of the same ID.
- Preserve streaming, per-Agent tool scope, interrupt repair, and context protocol validity.
- Route side-effecting operations through the harness safety boundary introduced by the active phase.
- Add focused tests for every behavior change.

## Workflow

- Work with existing user changes and keep edits scoped to the active phase.
- Before Phase 0, compile `corecoder`; after the package rename, compile `pikacore`.
- Run `uv lock --check`, `uv run --extra dev ruff check .`, `uv run --extra dev pytest tests/ -q`, and compileall before completing a phase.
- Do not commit or push unless the user explicitly requests it.

## Safety

- Never commit `.env`, `.pikacore/`, API keys, sessions, traces, or real benchmark prompts.
- Keep `uv.lock` tracked; it is not a generated artifact to ignore.
