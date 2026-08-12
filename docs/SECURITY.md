# PikaCore security boundary

[English](SECURITY.md) | [简体中文](SECURITY_CN.md)

The English documentation is canonical. Functional documentation changes must update
the English and Simplified Chinese versions in the same pull request.

PikaCore executes model-selected tools on the user's machine. Its controls reduce
accidental workspace escape, unapproved mutation, and credential leakage, but they do
not turn an untrusted model response or shell command into sandboxed code.

## Trust model

The user chooses the repository, model provider, prompt, permission mode, and approval
decisions. The configured model provider receives conversation content and selected
tool output. Local tools execute with the privileges of the PikaCore process.

Use `read-only` for inspection of untrusted repositories. Use the default `ask` mode
when changes are expected and inspect every proposed side effect. Use `auto` only when
the task, repository, provider, and local environment are trusted.

## Implemented controls

### Workspace paths

File-tool paths are resolved against the canonical workspace root. The resolver rejects:

- `..` or absolute paths that resolve outside the workspace;
- existing symlinks that escape the workspace;
- missing write targets whose existing parent resolves outside the workspace;
- the repository root itself as a file write target.

The boundary applies to path-aware file tools. It does not confine the operating-system
process created by the shell tool.

### Permissions and scheduling

Read-only tools are always allowed. In `read-only` mode, mutating/high-risk tools are
denied. In `ask`, the approval callback runs on the main thread before execution. In
`auto`, side-effecting tools run without approval.

Read-only batches may run concurrently. Writes, edits, shell calls, and sub-agent calls
are serialized barriers. Results are returned in the model's original call order.

### Shell execution

The shell tool:

- starts in the active workspace and tracks cwd per tool instance;
- reports the real process exit code;
- blocks a narrow list of clearly destructive patterns;
- truncates oversized display output while retaining head and tail;
- builds the child environment from an explicit cross-platform allowlist.

The allowlist includes normal process/runtime variables such as `PATH`, locale, temp,
home, and required Windows variables. Names containing `KEY`, `TOKEN`, `SECRET`,
`PASSWORD`, or `CREDENTIAL` are removed, so provider API keys are not inherited by
shell children.

The command-pattern filter is defense in depth, not a general shell policy. Equivalent
or obfuscated commands may bypass it. `shell=True` uses the platform's normal shell
semantics: POSIX shell on Unix-like systems and the Windows command processor on
Windows. Prompts and tests should not assume POSIX-only syntax when portability matters.

### Persistence and redaction

Project state lives under `.pikacore/` and is ignored by Git. JSON state uses atomic
replacement; JSONL trace lines are appended and flushed. Before persistence PikaCore
recursively redacts:

- values whose field names look like keys, tokens, secrets, passwords, or credentials;
- common bearer and API-key strings;
- credentials embedded in URLs.

Trace strings are limited to 4,000 characters. Session content is not truncated, so
resumed protocol messages remain complete after redaction. Redaction is best-effort and
state files are plaintext, not encrypted.

### Recovery

Assistant tool calls are checkpointed before side effects. If the required checkpoint
cannot be written, following mutation is stopped. Resume validates runtime identity and
file freshness. No saved tool call is replayed automatically. An unmatched call receives
one interrupted result, and recovery asks for workspace inspection before any retry.

## Out of scope

PikaCore does not currently provide:

- process, container, VM, or network isolation;
- filesystem confinement for arbitrary shell commands;
- encrypted state at rest;
- provider-side data-retention guarantees;
- a complete malware or destructive-command detector;
- protection from a user approving a harmful action;
- secret scanning for every possible credential format.

Sub-agents inherit the same workspace, model, and permission policy. They cannot spawn
another sub-agent, but their non-read-only work is still governed by the selected mode.

## Operational guidance

- Run PikaCore from the intended repository root and check `/session` before resuming.
- Keep the default `ask` mode unless unattended mutation is explicitly acceptable.
- Review `/permissions` and tool risk classifications after a resume or model switch.
- Treat `.pikacore/`, terminal scrollback, and `~/.pikacore_history` as potentially
  sensitive local data.
- Do not commit `.env`, `.pikacore/`, traces, sessions, or benchmark prompts containing
  real secrets.
- Inspect `git status` and `/diff` after interruption or any `incomplete-tool-call`
  recovery result.

## Reporting vulnerabilities

Use the repository's private vulnerability-reporting channel if one is enabled. If no
private channel is available, open a minimal GitHub issue without including secrets,
exploit payloads, or private repository content.
