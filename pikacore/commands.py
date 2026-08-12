"""Pure command routing for the interactive CLI state window."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass

from .agent import Agent
from .config import Config
from .session import list_sessions
from .state import SchemaMismatchError
from .working_memory import render_working_memory

ConfirmCallback = Callable[[str], bool]

HELP_TEXT = """Commands:
  /help                         Show this help
  /reset                        Clear conversation and Working Memory
  /memory                       Show Working Memory summary
  /memory files                 Show remembered files and freshness
  /memory clear                 Clear Working Memory after confirmation
  /session                      Show the active session
  /session list                 List saved sessions
  /session new                  Start a new session
  /session resume <id>          Validate and resume a session
  /sessions                     Alias for /session list
  /runs [n]                     Show recent runs (default: 10)
  /trace [run_id] [n]           Show recent trace events (default: 20)
  /permissions                  Show permission mode and tool risks
  /permissions <mode>           Set read-only, ask, or auto
  /tokens                       Show token usage
  /model [name]                 Show or switch model
  /compact                      Compress context through Agent durability
  /diff                         Show modified paths for this session
  /save [name]                  Save a named session snapshot
  quit                          Exit PikaCore"""


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    lines: tuple[str, ...] = ()
    exit_requested: bool = False


def execute_command(
    text: str,
    *,
    agent: Agent,
    config: Config,
    confirm: ConfirmCallback | None = None,
) -> CommandResult:
    """Parse and execute one local command without reading terminal input."""
    stripped = text.strip()
    if not stripped.startswith("/") and stripped.lower() not in {"quit", "exit"}:
        return CommandResult(handled=False)
    try:
        parts = shlex.split(stripped)
    except ValueError as exc:
        return _result(f"Invalid command syntax: {exc}")
    if not parts:
        return CommandResult(handled=False)

    command = parts[0].lower()
    arguments = parts[1:]
    if command in {"quit", "exit", "/quit", "/exit"}:
        return CommandResult(handled=True, exit_requested=True)
    if command == "/help":
        return _result(HELP_TEXT)
    if command == "/reset":
        if arguments:
            return _result("Usage: /reset")
        agent.reset()
        return _result("Conversation and Working Memory reset.")
    if command == "/memory":
        return _memory_command(agent, arguments, confirm)
    if command in {"/session", "/sessions"}:
        if command == "/sessions":
            arguments = ["list", *arguments]
        return _session_command(agent, config, arguments)
    if command == "/runs":
        return _runs_command(agent, arguments)
    if command == "/trace":
        return _trace_command(agent, arguments)
    if command == "/permissions":
        return _permissions_command(agent, arguments)
    if command == "/tokens":
        prompt_tokens, completion_tokens, cost = agent.session_token_usage()
        line = (
            f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion = "
            f"{prompt_tokens + completion_tokens} total"
        )
        if cost is not None:
            line += f" (~${cost:.4f})"
        return _result(line)
    if command == "/model":
        if len(arguments) > 1:
            return _result("Usage: /model [name]")
        if not arguments:
            return _result(f"Current model: {getattr(agent.llm, 'model', config.model)}")
        model = agent.set_model(arguments[0])
        config.model = model
        return _result(f"Switched to {model}.")
    if command == "/compact":
        if arguments:
            return _result("Usage: /compact")
        compression = agent.compact_context()
        if compression.changed:
            return _result(
                f"Compressed ({compression.strategy}): "
                f"{compression.before_tokens} -> {compression.after_tokens} tokens "
                f"({len(agent.messages)} messages)."
            )
        return _result(
            f"Nothing to compress ({compression.before_tokens} tokens, "
            f"{len(agent.messages)} messages)."
        )
    if command == "/diff":
        if arguments:
            return _result("Usage: /diff")
        paths = agent.modified_paths()
        if not paths:
            return _result("No files modified this session.")
        return _result(
            f"Files modified this session ({len(paths)}):",
            *(f"- {path}" for path in paths),
        )
    if command == "/save":
        if len(arguments) > 1:
            return _result("Usage: /save [name]")
        session_id = agent.save_session_snapshot(
            arguments[0] if arguments else None
        )
        return _result(
            f"Session saved: {session_id}",
            f"Resume with: pikacore -r {session_id}",
        )
    return _result(f"Unknown command: {parts[0]} (try /help)")


def _memory_command(
    agent: Agent,
    arguments: list[str],
    confirm: ConfirmCallback | None,
) -> CommandResult:
    memory = agent.session_state.working_memory
    if not arguments:
        rendered = render_working_memory(memory)
        lines = []
        if memory.current_request:
            lines.append(f"Current request: {memory.current_request}")
        if rendered:
            lines.append(rendered)
        return _result(*(lines or ["Working Memory is empty."]))
    if arguments == ["files"]:
        if not memory.files:
            return _result("Working Memory has no files.")
        return _result(*(
            f"{item.path}: action={item.action}, "
            f"fresh={'yes' if item.fresh else 'no'}, updated={item.updated_at}"
            for item in memory.files
        ))
    if arguments == ["clear"]:
        approved = bool(confirm and confirm(
            "Clear Working Memory? Conversation messages will be kept."
        ))
        if not approved:
            return _result("Working Memory clear cancelled.")
        agent.clear_working_memory()
        return _result("Working Memory cleared; conversation messages were kept.")
    return _result("Usage: /memory [files|clear]")


def _session_command(
    agent: Agent,
    config: Config,
    arguments: list[str],
) -> CommandResult:
    if not arguments:
        state = agent.session_state
        return _result(
            f"Session: {state.session_id}",
            f"Repo: {state.repo_root}",
            f"Model: {state.model}",
            f"Messages: {len(state.messages)}; runs: {len(state.run_ids)}",
            f"Saved: {state.updated_at}",
        )
    if arguments == ["list"]:
        sessions = list_sessions()
        if not sessions:
            return _result("No saved sessions.")
        return _result(*(
            f"{item['id']} ({item['model']}, {item['saved_at']}) {item['preview']}"
            for item in sessions
        ))
    if arguments == ["new"]:
        state = agent.new_session()
        config.model = state.model
        return _result(f"New session: {state.session_id}")
    if len(arguments) == 2 and arguments[0] == "resume":
        session_id = arguments[1]
        try:
            recovery = agent.resume_session(session_id)
        except SchemaMismatchError as exc:
            return _result(f"Cannot resume {session_id}: {exc.error_code}.")
        except (OSError, ValueError) as exc:
            return _result(f"Cannot resume {session_id}: {exc}.")
        if recovery is None:
            return _result(f"Session not found: {session_id}")
        if not recovery.can_resume:
            return _result(f"Cannot resume {session_id}: {recovery.status}.")
        config.model = getattr(agent.llm, "model", config.model)
        lines = [f"Resumed {session_id}: recovery={recovery.status}."]
        if recovery.notice:
            lines.append(recovery.notice)
        return _result(*lines)
    return _result("Usage: /session [list|new|resume <id>]")


def _runs_command(agent: Agent, arguments: list[str]) -> CommandResult:
    limit = _parse_limit(arguments, default=10, usage="/runs [n]")
    if isinstance(limit, CommandResult):
        return limit
    runs = agent.recent_runs(limit)
    if not runs:
        return _result("No runs for this session.")
    return _result(*(
        f"{run.run_id}: {run.status}, stop={run.stop_reason or '-'}, "
        f"model_attempts={run.model_attempts}, tool_steps={run.tool_steps}, "
        f"started={run.started_at}"
        for run in runs
    ))


def _trace_command(agent: Agent, arguments: list[str]) -> CommandResult:
    if len(arguments) > 2:
        return _result("Usage: /trace [run_id] [n]")
    run_id = None
    limit = 20
    if len(arguments) == 1:
        if arguments[0].isdigit():
            limit = _positive_int(arguments[0]) or 0
        else:
            run_id = arguments[0]
    elif len(arguments) == 2:
        run_id = arguments[0]
        limit = _positive_int(arguments[1]) or 0
    if limit <= 0:
        return _result("Usage: /trace [run_id] [n]")
    try:
        events, warnings = agent.recent_trace(run_id, limit)
    except (OSError, ValueError) as exc:
        return _result(f"Cannot read trace: {exc}.")
    if not events and not warnings:
        return _result("No trace events found.")
    lines = [
        f"#{event.seq} {event.timestamp} {event.event} {_brief_data(event.data)}"
        for event in events
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return _result(*lines)


def _permissions_command(agent: Agent, arguments: list[str]) -> CommandResult:
    if not arguments:
        tools = sorted(agent.tools, key=lambda item: item.name)
        return _result(
            f"Permission mode: {agent.permission_policy.mode}",
            *(
                f"{tool.name}: risk={tool.risk_level}, "
                f"scope={'read-only' if tool.read_only else 'mutating'}"
                for tool in tools
            ),
        )
    if len(arguments) != 1 or arguments[0] not in {"read-only", "ask", "auto"}:
        return _result("Usage: /permissions [read-only|ask|auto]")
    policy = agent.set_permission_mode(arguments[0])
    return _result(f"Permission mode changed to {policy.mode}.")


def _parse_limit(
    arguments: list[str],
    *,
    default: int,
    usage: str,
) -> int | CommandResult:
    if not arguments:
        return default
    if len(arguments) != 1:
        return _result(f"Usage: {usage}")
    value = _positive_int(arguments[0])
    return value if value is not None else _result(f"Usage: {usage}")


def _positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _brief_data(data: dict, limit: int = 500) -> str:
    rendered = ", ".join(f"{key}={data[key]!r}" for key in sorted(data))
    return rendered[:limit] + ("..." if len(rendered) > limit else "")


def _result(*lines: str) -> CommandResult:
    return CommandResult(handled=True, lines=tuple(lines))
