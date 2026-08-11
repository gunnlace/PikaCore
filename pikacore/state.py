"""Versioned persisted state models for sessions and individual runs."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

SCHEMA_VERSION = 1
RunStatus = Literal["running", "completed", "failed", "interrupted"]
StopReason = Literal[
    "completed",
    "max_rounds",
    "user_interrupted",
    "tool_rejected",
    "model_error",
    "internal_error",
]


class SchemaMismatchError(ValueError):
    """Raised when persisted state uses an unsupported schema version."""

    error_code = "schema-mismatch"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _validate_schema(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Persisted state must be a JSON object")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION}"
        )


class PersistedState:
    """Small serialization mixin shared by versioned state dataclasses."""

    _fields: ClassVar[tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        _validate_schema(data)
        return cls(**{name: data[name] for name in cls._fields if name in data})


@dataclass
class SessionState(PersistedState):
    schema_version: int = SCHEMA_VERSION
    session_id: str = field(default_factory=lambda: new_id("session"))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    repo_root: str = ""
    model: str = "unknown"
    messages: list[dict] = field(default_factory=list)
    # Phase 5 owns WorkingMemory behavior. Phase 3 only reserves its schema slot.
    working_memory: dict = field(default_factory=dict)
    last_checkpoint_id: str | None = None
    run_ids: list[str] = field(default_factory=list)

    _fields = (
        "schema_version",
        "session_id",
        "created_at",
        "updated_at",
        "repo_root",
        "model",
        "messages",
        "working_memory",
        "last_checkpoint_id",
        "run_ids",
    )

    def touch(self) -> None:
        self.updated_at = utc_now()


@dataclass
class RunState(PersistedState):
    schema_version: int = SCHEMA_VERSION
    run_id: str = field(default_factory=lambda: new_id("run"))
    session_id: str = ""
    user_request: str = ""
    status: RunStatus = "running"
    stop_reason: StopReason | None = None
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    model_attempts: int = 0
    tool_steps: int = 0
    final_answer: str | None = None
    error: str | None = None

    _fields = (
        "schema_version",
        "run_id",
        "session_id",
        "user_request",
        "status",
        "stop_reason",
        "started_at",
        "finished_at",
        "model_attempts",
        "tool_steps",
        "final_answer",
        "error",
    )


@dataclass
class Checkpoint(PersistedState):
    schema_version: int = SCHEMA_VERSION
    checkpoint_id: str = field(default_factory=lambda: new_id("checkpoint"))
    parent_checkpoint_id: str | None = None
    session_id: str = ""
    run_id: str = ""
    current_user_request: str = ""
    completed_tool_call_ids: list[str] = field(default_factory=list)
    pending_tool_calls: list[dict] = field(default_factory=list)
    last_successful_action: str | None = None
    next_suggested_action: str | None = None
    file_freshness: dict[str, str] = field(default_factory=dict)
    runtime_identity: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    _fields = (
        "schema_version",
        "checkpoint_id",
        "parent_checkpoint_id",
        "session_id",
        "run_id",
        "current_user_request",
        "completed_tool_call_ids",
        "pending_tool_calls",
        "last_successful_action",
        "next_suggested_action",
        "file_freshness",
        "runtime_identity",
        "created_at",
    )


TRACE_EVENT_NAMES = frozenset({
    "run_started",
    "message_appended",
    "context_built",
    "model_requested",
    "model_completed",
    "tool_requested",
    "tool_approved",
    "tool_rejected",
    "tool_completed",
    "working_memory_updated",
    "context_compressed",
    "checkpoint_created",
    "run_finished",
    "run_failed",
})


@dataclass
class TraceEvent(PersistedState):
    schema_version: int = SCHEMA_VERSION
    seq: int = 0
    timestamp: str = field(default_factory=utc_now)
    event: str = "run_started"
    session_id: str = ""
    run_id: str = ""
    data: dict = field(default_factory=dict)

    _fields = (
        "schema_version",
        "seq",
        "timestamp",
        "event",
        "session_id",
        "run_id",
        "data",
    )

    def __post_init__(self) -> None:
        if self.event not in TRACE_EVENT_NAMES:
            raise ValueError(f"Unknown trace event: {self.event}")


@dataclass
class Report(PersistedState):
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    session_id: str = ""
    model: str = "unknown"
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int = 0
    model_attempts: int = 0
    tool_steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    tool_errors: dict[str, int] = field(default_factory=dict)
    tool_approvals: dict[str, int] = field(default_factory=dict)
    tool_error_count: int = 0
    approval_count: int = 0
    read_paths: list[str] = field(default_factory=list)
    affected_paths: list[str] = field(default_factory=list)
    context_compressions: int = 0
    context_tokens_before: int = 0
    context_tokens_after: int = 0
    checkpoint_status: str | None = None
    recovery_status: str | None = None
    stop_reason: StopReason | None = None
    completed: bool = False
    error: str | None = None
    persistence_errors: list[str] = field(default_factory=list)

    _fields = (
        "schema_version",
        "run_id",
        "session_id",
        "model",
        "started_at",
        "finished_at",
        "duration_ms",
        "model_attempts",
        "tool_steps",
        "prompt_tokens",
        "completion_tokens",
        "tool_calls",
        "tool_errors",
        "tool_approvals",
        "tool_error_count",
        "approval_count",
        "read_paths",
        "affected_paths",
        "context_compressions",
        "context_tokens_before",
        "context_tokens_after",
        "checkpoint_status",
        "recovery_status",
        "stop_reason",
        "completed",
        "error",
        "persistence_errors",
    )
