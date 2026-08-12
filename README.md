# PikaCore

PikaCore is a small coding-agent harness with native function calling, project-local
state, resumable runs, permission controls, Working Memory, context compression, and
an offline evaluation suite.

PikaCore is forked from [CoreCoder](https://github.com/he-yufeng/CoreCoder). The
original CoreCoder copyright and MIT license notice are preserved in
[LICENSE](LICENSE); PikaCore remains distributed under the MIT License.

## What is implemented

- Streaming OpenAI-compatible and optional LiteLLM model access.
- Native tool calls for file reads, writes, edits, search, shell commands, and a
  non-recursive sub-agent.
- Repository path boundaries for file tools, three permission modes, main-thread
  approval, sanitized shell environments, and structured tool results.
- Parallel read batches with serialized write, shell, and sub-agent barriers while
  preserving model-requested result order.
- Atomic session, run, checkpoint, and report JSON plus redacted JSONL traces under
  the current repository's `.pikacore/` directory.
- Five-way recovery classification, file freshness checks, and interrupted-result
  repair without automatically replaying pending tools.
- Bounded, event-driven Working Memory and observable layered context compression.
- Ten deterministic fixture benchmarks driven by `ScriptedFakeLLM`; the benchmark
  runner does not call a real provider by default.

See [Current architecture](docs/PIKACORE_DESIGN.md),
[Security boundary](docs/SECURITY.md), and [Benchmark results](docs/BENCHMARKS.md)
for the detailed contracts.

## Requirements

- Python 3.10–3.13
- `uv` for the documented install and development workflow
- An API key for interactive or one-shot model use. The offline benchmarks and test
  suite do not need one.

## Install from source

PikaCore is not configured for PyPI publishing. Install it from this repository:

```bash
git clone https://github.com/gunnlace/PikaCore.git
cd PikaCore
uv sync
uv run pikacore --version
```

For development tools:

```bash
uv sync --extra dev
```

For the optional LiteLLM backend:

```bash
uv sync --extra litellm
```

## Configure a model

PikaCore reads `PIKACORE_*` variables first. `OPENAI_API_KEY` and
`OPENAI_BASE_URL` retain their conventional meanings, and `CORECODER_*` names are
compatibility fallbacks.

| Setting | Resolution order |
|---|---|
| API key | `PIKACORE_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `CORECODER_API_KEY` |
| Model | `PIKACORE_MODEL`, `CORECODER_MODEL`, then `gpt-5.5` |
| Base URL | `PIKACORE_BASE_URL`, `OPENAI_BASE_URL`, `CORECODER_BASE_URL` |
| Output limit | `PIKACORE_MAX_TOKENS`, `CORECODER_MAX_TOKENS`, then `4096` |
| Context limit | `PIKACORE_MAX_CONTEXT`, `CORECODER_MAX_CONTEXT`, then `128000` |
| Temperature | `PIKACORE_TEMPERATURE`, `CORECODER_TEMPERATURE`, then `0` |
| Backend | `PIKACORE_PROVIDER`, `CORECODER_PROVIDER`, then `openai` |

```bash
export OPENAI_API_KEY=sk-...
export PIKACORE_MODEL=gpt-5.5
uv run pikacore
```

For another OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=your-key
export PIKACORE_BASE_URL=https://api.example.com/v1
export PIKACORE_MODEL=provider-model
uv run pikacore
```

After installing the `litellm` extra, select that backend with
`PIKACORE_PROVIDER=litellm`.

The corresponding CLI options are `--api-key`, `--base-url`, and `--model`.
Configuration precedence is CLI option, primary environment variable, compatibility
fallback, then the built-in default.

## Run PikaCore

Start the interactive REPL from the repository you want PikaCore to operate on:

```bash
uv run pikacore --permissions ask
```

Run one request and exit:

```bash
uv run pikacore --permissions read-only -p "Explain the failing tests"
```

Resume a saved session:

```bash
uv run pikacore -r session_id
```

PikaCore discovers the containing Git repository as its workspace. Outside Git, the
current directory becomes the workspace root.

## Permission modes

`ask` is the default.

| Mode | Read-only tools | Write, edit, shell, sub-agent |
|---|---|---|
| `read-only` | allowed | denied |
| `ask` | allowed | requires terminal approval |
| `auto` | allowed | allowed without approval |

Choose the initial mode with `--permissions read-only|ask|auto`, or inspect/change
the current process with `/permissions`. A runtime change is recorded in the run
trace and checkpoint identity.

`auto` is not a filesystem or process sandbox. Read [Security boundary](docs/SECURITY.md)
before enabling it on an untrusted task.

## Local commands

| Command | Behavior |
|---|---|
| `/help` | Show command help. |
| `/reset` | Clear conversation messages, Working Memory, and recovery continuity. |
| `/memory` | Show the current bounded Working Memory. |
| `/memory files` | Show remembered files, actions, and freshness. |
| `/memory clear` | Confirm, then clear Working Memory without clearing messages. |
| `/session` | Show active session metadata. |
| `/session list` | List recent project sessions. |
| `/session new` | Save the current session and create an empty one. |
| `/session resume <id>` | Validate recovery state and switch sessions. |
| `/sessions` | Compatibility alias for `/session list`. |
| `/runs [n]` | Show recent runs for this session; default 10. |
| `/trace [run_id] [n]` | Show recent redacted events; default current run and 20 events. |
| `/permissions [mode]` | Show tool risks or set `read-only`, `ask`, or `auto`. |
| `/tokens` | Aggregate token use and known-model cost from this session's reports. |
| `/model [name]` | Show or change the model and checkpoint runtime identity. |
| `/compact` | Run context compression through the durable Agent lifecycle. |
| `/diff` | Show paths attributed to tool results in this session. |
| `/save [name]` | Save a complete named SessionState snapshot. Quote names with spaces. |
| `quit` | Exit the REPL. |

Local commands call Agent and Store APIs; the CLI command router does not read or
write state JSON directly.

## State directory and recovery

Runtime state is scoped to the current repository:

```text
.pikacore/
├── sessions/<session_id>.json
├── runs/<run_id>/task_state.json
├── runs/<run_id>/trace.jsonl
├── runs/<run_id>/report.json
├── checkpoints/<checkpoint_id>.json
└── benchmarks/phase6-report.json   # created when benchmarks run
```

`.pikacore/` is ignored by Git. REPL input history is stored separately at
`~/.pikacore_history` and is also ignored by this repository.

On resume, PikaCore compares the checkpoint with the current model, repository,
branch, tool schema, permission mode, and file fingerprints. Recovery is classified
as `full-valid`, `files-stale`, `runtime-mismatch`, `incomplete-tool-call`, or
`schema-mismatch`. Pending tool calls are never replayed automatically. Orphan calls
receive an interrupted tool result, and mutating or otherwise unknown prior work is
accompanied by a workspace-inspection notice. See
[Current architecture](docs/PIKACORE_DESIGN.md#recovery-semantics).

## Offline benchmarks

Run the baseline and both supported ablations:

```bash
uv run python benchmarks/run_benchmarks.py --ablation all
```

The current deterministic suite has 10 fixtures and 3 variants. On 2026-08-12 the
baseline passed 10/10; Working Memory off passed 9/10; context policy off passed
9/10. All 30 outcomes completed, for 28/30 passing checks. The two ablation failures
are the intended feature-sensitivity checks. Full methodology and per-variant results
are in [Benchmark results](docs/BENCHMARKS.md).

## Development checks

```bash
uv lock --check
uv run --extra dev ruff check .
uv run --extra dev pytest tests/ -q
uv run python -m compileall pikacore
```

## Security summary

File tools reject `..` traversal, absolute paths outside the workspace, and symlink
escapes. Shell subprocesses receive an allowlisted environment that excludes API
keys and other credential-like variables. State and traces are recursively redacted
before persistence, with trace strings capped at 4,000 characters.

These controls reduce accidental leakage and cross-repository writes; they are not
an OS sandbox. Shell commands can still access resources available to the current
user, model requests send selected context to the configured provider, and local
state is not encrypted. Use `read-only` or `ask` for untrusted work and keep
`.pikacore/` private. Read [Security boundary](docs/SECURITY.md) for the complete
boundary.

## License and upstream attribution

PikaCore is a fork of [CoreCoder](https://github.com/he-yufeng/CoreCoder), originally
copyright © 2026 Yufeng He. The upstream MIT copyright and permission notice are
preserved verbatim in [LICENSE](LICENSE). PikaCore modifications are provided under
the same MIT License.
