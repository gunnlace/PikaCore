# PikaCore current architecture

[English](PIKACORE_DESIGN.md) | [简体中文](PIKACORE_DESIGN_CN.md)

The English documentation is canonical. Functional documentation changes must update
the English and Simplified Chinese versions in the same pull request.

This document describes the behavior implemented in PikaCore 0.1.0. It is an
implementation reference, not a roadmap.

PikaCore is a fork of [CoreCoder](https://github.com/he-yufeng/CoreCoder). It keeps
CoreCoder's compact Python package and native function-calling loop while adding a
workspace boundary, permission-aware tool execution, durable state, recovery,
Working Memory, context compression, CLI state views, and offline fixture benchmarks.
The original CoreCoder MIT attribution remains in the repository's `LICENSE` file.

## Architecture

```text
CLI / one-shot prompt
        │
        ▼
Agent ───────────────► LLM / LiteLLM
  │                       │
  │ native tool calls     │ streamed text + structured calls
  ▼                       │
ToolExecutor ◄────────────┘
  │
  ├── PermissionPolicy + main-thread approval
  ├── read batches in parallel
  └── write / bash / agent barriers in order
        │
        ▼
WorkspaceContext + built-in tools
        │
        ├── structured ToolExecutionResult
        └── file freshness / workspace delta

Agent durability points
  ├── ProjectStore: SessionState, RunState, Checkpoint, Report, TraceEvent
  ├── WorkingMemoryManager: structured runtime events only
  └── ContextManager: protocol-safe layered compression
```

### Component responsibilities

| Component | Implemented responsibility |
|---|---|
| `cli.py` | CLI arguments, REPL rendering, terminal approval, streaming display. |
| `commands.py` | Pure local-command parsing and calls to Agent/Store-backed APIs. |
| `Agent` | Model/tool loop, tool-call pairing, run lifecycle, durability, recovery wiring. |
| `ToolExecutor` | Tool lookup, argument binding, permission decisions, approval, scheduling, structured results. |
| `WorkspaceContext` | Git-root discovery, canonical path resolution, workspace snapshots and fingerprints. |
| `ProjectStore` | Atomic JSON, redacted JSONL, schema loading, complete named session snapshots. |
| `WorkingMemoryManager` | Deterministic bounded updates from user, tool, checkpoint, recovery, and run events. |
| `ContextManager` | Token estimation, safe protocol splits, layered compression and `CompressionResult`. |
| `benchmarks/` | Isolated fixture repositories, `ScriptedFakeLLM`, ablations, deterministic outcome view, JSON report. |

## Agent and tool protocol

Each `Agent` owns its tool instances and its tool-name lookup table. An assistant tool
call must be followed by exactly one tool result with the same ID. Parallel tool
execution does not change the result order seen by the model.

The built-in tool set is:

- `read_file`
- `write_file`
- `edit_file`
- `glob`
- `grep`
- `bash`
- `agent`

Consecutive read-only calls run as a parallel batch. Every write, edit, shell, or
sub-agent call forms a serialized barrier. A rejected barrier also rejects following
barrier calls in the same group. Tool completion is normalized into
`ToolExecutionResult`, including status, error code, approval outcome, exit code,
read paths, affected paths, workspace-change flag, truncation, and duration.

The sub-agent has a separate conversation and run identity, shares the parent model,
workspace, store, and permission policy, and cannot receive the `agent` tool. This is
the implemented recursion guard.

## Workspace and permission boundary

The workspace is the containing Git repository; if Git discovery fails, it is the
starting directory. File arguments are canonicalized before execution. The resolver
rejects traversal, paths outside the root, symlink escapes, and use of the workspace
root itself as a write target. Missing write targets are accepted only after their
existing parent chain resolves inside the workspace.

Permission modes are evaluated per tool:

| Mode | Read-only tool | Mutating/high-risk tool |
|---|---|---|
| `read-only` | allow | deny |
| `ask` | allow | invoke the CLI approval callback on the main thread |
| `auto` | allow | allow |

`bash` is classified as high-risk and mutating. It blocks a small set of known
destructive command patterns, starts in the workspace, maintains cwd per BashTool
instance, captures the actual exit code, and truncates very large display output.
Its subprocess environment is an explicit cross-platform allowlist and excludes
credential-like variables, including API keys.

These application controls are not an operating-system sandbox. The precise boundary
and limitations are documented in [SECURITY.md](SECURITY.md).

## Durable state

State schema version 1 is stored below the current project:

```text
.pikacore/
├── sessions/<session_id>.json
├── runs/<run_id>/task_state.json
├── runs/<run_id>/trace.jsonl
├── runs/<run_id>/report.json
└── checkpoints/<checkpoint_id>.json
```

`SessionState` owns structured messages, Working Memory, model, repository root,
checkpoint link, and run IDs. `RunState` tracks one user request or state-changing CLI
operation. `Report` freezes model/tool/token/compression/approval/path metrics when a
run ends. `TraceEvent` records the fixed event vocabulary in append-only JSONL.

Session, run, checkpoint, and report files use atomic replacement. Trace lines are
appended and flushed. Recursive secret redaction is applied before persistence;
Session strings are not length-truncated, while trace/report summaries use bounded
strings. A corrupt final trace line is ignored with a warning, but corruption in an
earlier line is reported as an error.

Named `/save` snapshots copy the complete `SessionState` and clone the linked
checkpoint with the snapshot's new session identity. The CLI never opens these files
directly.

### Durability points

The Agent persists at message boundaries and around operations needed for recovery.
Assistant tool calls are checkpointed before any tool side effect. Tool results are
saved and paired before the loop proceeds. Read-batch completion and mutating barriers
create checkpoints. Manual context compaction and model/permission changes run through
the same run/report/checkpoint lifecycle.

If a required checkpoint cannot be saved, the Agent does not start the following
side-effecting tool. Persistence failures are surfaced as warnings and, when a report
can be written, included in report metrics.

## Recovery semantics

Resume is available through `-r/--resume` and `/session resume <id>`. The current
runtime identity includes:

- model;
- canonical repository root;
- current Git branch;
- tool names, risk/read-only flags, and schemas as one signature;
- permission mode;
- state schema version.

Checkpoint file freshness stores hashes for relevant read or modified paths.
Directory-style `grep` and `glob` searches are conservatively marked unverifiable so
recovery requests a recheck when the result set could have changed.

Recovery returns one of five classifications:

| Status | Meaning and behavior |
|---|---|
| `full-valid` | Runtime identity and saved file fingerprints still match. |
| `files-stale` | A file changed, disappeared, or a directory search is unverifiable; append a notice and require rereading. |
| `runtime-mismatch` | Model, root, branch, tools, permission mode, or checkpoint presence differs; append a review notice. |
| `incomplete-tool-call` | An assistant call has no result; add exactly one interrupted result and require workspace inspection. |
| `schema-mismatch` | Persisted schema is unsupported; do not resume or mutate the saved state. |

No pending tool call is automatically replayed. This includes reads as well as
writes. Unknown prior write, edit, shell, or sub-agent execution is represented as
interrupted; the recovery notice instructs the model and user to inspect the workspace
before deciding whether to retry.

## Working Memory

Working Memory is part of `SessionState`; there is no separate memory directory. It
contains the current request, a compact task summary, remembered files and freshness,
10 recent shell commands, up to 10 blockers, and up to 10 next steps. File memory is
limited to 30 entries. Task/file/item strings also have explicit length bounds.

Updates come only from structured user, tool, checkpoint, recovery, and run events.
Final-answer prose is never parsed for magic phrases. A successful reread refreshes a
file hash and resolves its associated stale-file blocker. `/memory clear` clears only
Working Memory after confirmation; `/reset` clears messages, Working Memory, and
recovery continuity.

## Context compression

`ContextManager` estimates token pressure and applies implemented layers in this
order:

1. snip old tool output;
2. merge duplicate reads;
3. extract old search and command material;
4. summarize old turns through the configured LLM when needed;
5. collapse older turns under hard pressure.

Tool-call/result groups remain structured, and `_safe_split` avoids cutting an active
protocol group. Working Memory and old turns receive separate summary-input budgets so
large memory cannot exclude all conversation history. Every changed compression emits
a `CompressionResult` and records strategy, before/after token estimates, and message
counts in trace/report state. `/compact` uses this Agent-level durable path.

## CLI state window

The local command router exposes stable Agent and Store APIs rather than decoding JSON.
Implemented commands are:

```text
/help
/reset
/memory
/memory files
/memory clear
/session
/session list
/session new
/session resume <id>
/sessions
/runs [n]
/trace [run_id] [n]
/permissions [read-only|ask|auto]
/tokens
/model [name]
/compact
/diff
/save [name]
quit
```

`/tokens` aggregates report totals for the active session. `/diff` uses paths attributed
to structured tool results and reports, scoped to the active session. Command parsing
has no model call and accepts an injected confirmation callback for testability.

## Offline evaluation

The evaluation harness materializes only manifest-listed regular fixture files in a
temporary directory. `ScriptedFakeLLM` supplies deterministic responses; timestamps,
durations, UUIDs, and other non-deterministic fields are excluded from outcome
comparison. The only supported variants are baseline, Working Memory off, and context
policy off.

The implemented fixtures cover edit success, bad-argument retry, path escape,
permission rejection, parallel reads, write barriers, large output, stale-file resume,
unknown-write resume, and Working Memory freshness. Current measured results and the
runner command are in [BENCHMARKS.md](BENCHMARKS.md).
