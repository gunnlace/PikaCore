"""Atomic project-local storage for Phase 3 runtime artifacts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security import redact
from .state import (
    Checkpoint,
    Report,
    RunState,
    SessionState,
    TraceEvent,
    new_id,
    utc_now,
)

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class TraceReadResult:
    events: list[TraceEvent]
    warnings: list[str]


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Replace a JSON file atomically after flushing its complete contents."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, data: Any) -> None:
    """Append one redacted JSON object and flush it before returning."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(redact(data), ensure_ascii=False, sort_keys=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()


class ProjectStore:
    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        state_root: str | Path | None = None,
    ):
        root = Path(repo_root or Path.cwd()).expanduser().resolve()
        self.state_root = (
            Path(state_root).expanduser().resolve()
            if state_root is not None
            else root / ".pikacore"
        )

    @property
    def sessions_dir(self) -> Path:
        return self.state_root / "sessions"

    @property
    def runs_dir(self) -> Path:
        return self.state_root / "runs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.state_root / "checkpoints"

    def session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{_safe_component(session_id)}.json"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / _safe_component(run_id)

    def task_state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "task_state.json"

    def trace_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "trace.jsonl"

    def report_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "report.json"

    def checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.checkpoints_dir / f"{_safe_component(checkpoint_id)}.json"

    def save_session(self, state: SessionState) -> None:
        atomic_write_json(
            self.session_path(state.session_id),
            redact(state.to_dict(), max_string_length=None),
        )

    def load_session(self, session_id: str) -> SessionState | None:
        path = self.session_path(session_id)
        if not path.exists():
            return None
        return SessionState.from_dict(read_json(path))

    def save_session_snapshot(
        self,
        state: SessionState,
        name: str | None = None,
    ) -> SessionState:
        """Save a complete session branch, including a valid checkpoint link."""
        snapshot = deepcopy(state)
        snapshot.session_id = _snapshot_id(name)
        timestamp = utc_now()
        snapshot.created_at = timestamp
        snapshot.updated_at = timestamp

        if snapshot.last_checkpoint_id is not None:
            checkpoint = self.load_checkpoint(snapshot.last_checkpoint_id)
            if checkpoint is not None:
                checkpoint = deepcopy(checkpoint)
                checkpoint.parent_checkpoint_id = checkpoint.checkpoint_id
                checkpoint.checkpoint_id = new_id("checkpoint")
                checkpoint.session_id = snapshot.session_id
                checkpoint.created_at = timestamp
                self.save_checkpoint(checkpoint)
                snapshot.last_checkpoint_id = checkpoint.checkpoint_id

        self.save_session(snapshot)
        return snapshot

    def save_run(self, state: RunState) -> None:
        atomic_write_json(
            self.task_state_path(state.run_id),
            redact(state.to_dict(), max_string_length=None),
        )

    def load_run(self, run_id: str) -> RunState | None:
        path = self.task_state_path(run_id)
        if not path.exists():
            return None
        return RunState.from_dict(read_json(path))

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        data = checkpoint.to_dict()
        # file_freshness keys are workspace paths, not secret field names. A
        # path such as tokenizer.py must retain its SHA-256 value verbatim.
        file_freshness = dict(data.pop("file_freshness", {}))
        persisted = redact(data, max_string_length=None)
        persisted["file_freshness"] = file_freshness
        atomic_write_json(
            self.checkpoint_path(checkpoint.checkpoint_id),
            persisted,
        )

    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        path = self.checkpoint_path(checkpoint_id)
        if not path.exists():
            return None
        return Checkpoint.from_dict(read_json(path))

    def append_trace(self, event: TraceEvent) -> None:
        append_jsonl(self.trace_path(event.run_id), event.to_dict())

    def read_trace(self, run_id: str) -> TraceReadResult:
        path = self.trace_path(run_id)
        if not path.exists():
            return TraceReadResult([], [])

        events: list[TraceEvent] = []
        warnings: list[str] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                if index != len(lines) - 1:
                    raise ValueError(f"Corrupt trace line {index + 1}: {exc}") from exc
                warnings.append(f"Ignored corrupt final trace line {index + 1}: {exc}")
                continue
            if not isinstance(data, dict):
                raise ValueError(f"Trace line {index + 1} is not a JSON object")
            events.append(TraceEvent.from_dict(data))
        return TraceReadResult(events, warnings)

    def save_report(self, report: Report) -> None:
        atomic_write_json(self.report_path(report.run_id), redact(report.to_dict()))

    def load_report(self, run_id: str) -> Report | None:
        path = self.report_path(run_id)
        if not path.exists():
            return None
        return Report.from_dict(read_json(path))


def _safe_component(value: str) -> str:
    if not value or not _SAFE_COMPONENT_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError("Invalid state identifier")
    return value


def _snapshot_id(name: str | None) -> str:
    if not name:
        return new_id("session")
    normalized = name.strip().replace("\\", "/").split("/")[-1]
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized).strip(".-_")
    normalized = normalized[:100].strip(".-_")
    return normalized or new_id("session")
