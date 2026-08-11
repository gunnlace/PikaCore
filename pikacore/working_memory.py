"""Structured, bounded Working Memory for one PikaCore session."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .checkpoint import RecoveryResult, UNVERIFIABLE_FRESHNESS
from .state import CommandMemory, FileMemory, WorkingMemory
from .tool_executor import ToolExecutionResult
from .workspace import WorkspaceContext

MAX_FILES = 30
MAX_COMMANDS = 10
MAX_BLOCKERS = 10
MAX_NEXT_STEPS = 10
MAX_TASK_SUMMARY_CHARS = 400
MAX_FILE_SUMMARY_CHARS = 600
MAX_ITEM_CHARS = 400


@dataclass(frozen=True)
class UserMemoryEvent:
    request: str
    run_id: str
    occurred_at: str


@dataclass(frozen=True)
class ToolMemoryEvent:
    tool_name: str
    arguments: dict
    result: ToolExecutionResult
    run_id: str
    occurred_at: str


@dataclass(frozen=True)
class CheckpointMemoryEvent:
    file_freshness: dict[str, str]
    pending_tool_names: list[str]
    next_suggested_action: str | None
    occurred_at: str


@dataclass(frozen=True)
class RecoveryMemoryEvent:
    result: RecoveryResult
    occurred_at: str


@dataclass(frozen=True)
class RunMemoryEvent:
    status: str
    stop_reason: str
    run_id: str
    occurred_at: str


MemoryEvent = (
    UserMemoryEvent
    | ToolMemoryEvent
    | CheckpointMemoryEvent
    | RecoveryMemoryEvent
    | RunMemoryEvent
)


class WorkingMemoryManager:
    """Apply validated runtime events without interpreting assistant prose."""

    def __init__(self, memory: WorkingMemory, workspace: WorkspaceContext):
        self.memory = memory
        self.workspace = workspace

    def apply(self, event: MemoryEvent) -> bool:
        if isinstance(event, UserMemoryEvent):
            changed = self._apply_user(event)
        elif isinstance(event, ToolMemoryEvent):
            changed = self._apply_tool(event)
        elif isinstance(event, CheckpointMemoryEvent):
            changed = self._apply_checkpoint(event)
        elif isinstance(event, RecoveryMemoryEvent):
            changed = self._apply_recovery(event)
        elif isinstance(event, RunMemoryEvent):
            # Completion is intentionally structural only. In particular, the
            # final answer is not part of this event and cannot update memory.
            changed = False
        else:  # pragma: no cover - defensive for callers bypassing the type API
            raise TypeError(f"Unsupported Working Memory event: {type(event).__name__}")

        if changed:
            self.memory.updated_at = event.occurred_at
            self._enforce_capacity()
        return changed

    def _apply_user(self, event: UserMemoryEvent) -> bool:
        summary = _compact(event.request, MAX_TASK_SUMMARY_CHARS)
        changed = (
            self.memory.current_request != event.request
            or self.memory.task_summary != summary
        )
        self.memory.current_request = event.request
        self.memory.task_summary = summary
        return changed

    def _apply_tool(self, event: ToolMemoryEvent) -> bool:
        changed = False
        result = event.result

        if event.tool_name == "read_file" and result.status == "ok":
            path = self._first_path(result.read_paths, event.arguments.get("file_path"))
            if path is not None:
                fingerprint = self._fingerprint(path)
                self._upsert_file(FileMemory(
                    path=path,
                    action="read",
                    summary=_compact(result.content, MAX_FILE_SUMMARY_CHARS),
                    content_hash=fingerprint,
                    fresh=fingerprint is not None,
                    updated_at=event.occurred_at,
                ))
                self._remove_value(self.memory.next_steps, f"Reread {path}")
                changed = True

        if event.tool_name in {"write_file", "edit_file"} and result.status == "ok":
            candidates = list(result.affected_paths)
            argument_path = event.arguments.get("file_path")
            if argument_path and not candidates:
                candidates.append(str(argument_path))
            for raw_path in candidates:
                path = self._relative_path(raw_path)
                if path is None:
                    continue
                existing = self._file_by_path(path)
                summary = (
                    existing.summary if existing is not None else "Modified; reread required."
                )
                self._upsert_file(FileMemory(
                    path=path,
                    action="modified",
                    summary=summary,
                    content_hash=self._fingerprint(path),
                    fresh=False,
                    updated_at=event.occurred_at,
                ))
                self._append_unique(self.memory.next_steps, f"Reread {path}")
                changed = True

        if event.tool_name == "bash":
            command = str(event.arguments.get("command", ""))
            self.memory.recent_commands.append(CommandMemory(
                command=_compact(command, MAX_ITEM_CHARS),
                exit_code=result.exit_code,
                status=result.status,
                run_id=event.run_id,
                executed_at=event.occurred_at,
            ))
            changed = True

        if result.status != "ok" or result.approval == "rejected":
            detail = result.error_code or result.status
            self._append_unique(
                self.memory.blockers,
                _compact(f"{event.tool_name}: {detail}", MAX_ITEM_CHARS),
            )
            changed = True
        return changed

    def _apply_checkpoint(self, event: CheckpointMemoryEvent) -> bool:
        changed = False
        pending_check = "inspect workspace before deciding whether to retry pending tools"
        if not event.pending_tool_names and pending_check in self.memory.next_steps:
            self._remove_value(self.memory.next_steps, pending_check)
            changed = True
        for item in self.memory.files:
            expected = event.file_freshness.get(item.path)
            if expected is None:
                continue
            fresh = (
                item.action == "read"
                and expected != UNVERIFIABLE_FRESHNESS
                and item.content_hash == expected
            )
            if item.fresh != fresh:
                item.fresh = fresh
                item.updated_at = event.occurred_at
                changed = True
        if event.next_suggested_action:
            self._append_unique(
                self.memory.next_steps,
                _compact(event.next_suggested_action, MAX_ITEM_CHARS),
            )
            changed = True
        return changed

    def _apply_recovery(self, event: RecoveryMemoryEvent) -> bool:
        result = event.result
        changed = False
        if result.status == "files-stale":
            for raw_path in result.stale_paths:
                path = self._relative_path(raw_path) or str(raw_path)
                item = self._file_by_path(path)
                if item is not None:
                    item.fresh = False
                    item.updated_at = event.occurred_at
                self._append_unique(self.memory.next_steps, f"Reread {path}")
            self._append_unique(
                self.memory.blockers,
                _compact(
                    f"Recovery found stale files: {', '.join(result.stale_paths)}",
                    MAX_ITEM_CHARS,
                ),
            )
            changed = True
        elif result.status == "runtime-mismatch":
            keys = ", ".join(sorted(result.runtime_differences))
            self._append_unique(
                self.memory.blockers,
                _compact(f"Recovery runtime mismatch: {keys}", MAX_ITEM_CHARS),
            )
            self._append_unique(
                self.memory.next_steps,
                _compact(f"Review runtime differences: {keys}", MAX_ITEM_CHARS),
            )
            changed = True
        elif result.status == "incomplete-tool-call":
            tools = ", ".join(result.pending_tool_names) or "pending tools"
            self._append_unique(
                self.memory.blockers,
                _compact(f"Recovery found incomplete tool calls: {tools}", MAX_ITEM_CHARS),
            )
            self._append_unique(
                self.memory.next_steps,
                _compact(
                    f"Inspect workspace before deciding whether to retry: {tools}",
                    MAX_ITEM_CHARS,
                ),
            )
            changed = True
        elif result.status == "schema-mismatch":
            self._append_unique(
                self.memory.blockers,
                "Recovery schema mismatch; this session cannot be resumed.",
            )
            changed = True
        return changed

    def _first_path(self, paths: list[str], fallback) -> str | None:
        candidates = list(paths)
        if fallback:
            candidates.append(str(fallback))
        for raw_path in candidates:
            relative = self._relative_path(raw_path)
            if relative is not None:
                return relative
        return None

    def _relative_path(self, path: str) -> str | None:
        try:
            resolved = self.workspace.resolve_path(path)
            return resolved.relative_to(self.workspace.repo_root).as_posix()
        except (OSError, ValueError):
            return None

    def _fingerprint(self, path: str) -> str | None:
        try:
            _, fingerprint = self.workspace.fingerprint_path(path)
            return fingerprint
        except (OSError, ValueError):
            return None

    def _file_by_path(self, path: str) -> FileMemory | None:
        return next((item for item in self.memory.files if item.path == path), None)

    def _upsert_file(self, item: FileMemory) -> None:
        self.memory.files = [entry for entry in self.memory.files if entry.path != item.path]
        self.memory.files.append(item)

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        values[:] = [item for item in values if item != value]
        values.append(value)

    @staticmethod
    def _remove_value(values: list[str], value: str) -> None:
        values[:] = [item for item in values if item != value]

    def _enforce_capacity(self) -> None:
        self.memory.files[:] = self.memory.files[-MAX_FILES:]
        self.memory.recent_commands[:] = self.memory.recent_commands[-MAX_COMMANDS:]
        self.memory.blockers[:] = self.memory.blockers[-MAX_BLOCKERS:]
        self.memory.next_steps[:] = self.memory.next_steps[-MAX_NEXT_STEPS:]


def render_working_memory(memory: WorkingMemory) -> str:
    """Render a short stable prompt section; current_request stays structured."""
    if not any((
        memory.task_summary,
        memory.files,
        memory.recent_commands,
        memory.blockers,
        memory.next_steps,
    )):
        return ""

    lines = ["[Working memory]"]
    if memory.task_summary:
        lines.append(f"Task summary: {memory.task_summary}")
    if memory.files:
        lines.append("Files:")
        for item in memory.files:
            freshness = "fresh" if item.fresh else "stale"
            lines.append(
                f"- {item.path} ({item.action}, {freshness}): {item.summary}"
            )
    if memory.recent_commands:
        lines.append("Recent commands:")
        for item in memory.recent_commands:
            lines.append(
                f"- [{item.status}, exit={item.exit_code}] {item.command}"
            )
    if memory.blockers:
        lines.append("Blockers:")
        lines.extend(f"- {item}" for item in memory.blockers)
    if memory.next_steps:
        lines.append("Next required checks:")
        lines.extend(f"- {item}" for item in memory.next_steps)
    return "\n".join(lines)


def _compact(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"
