"""Checkpoint runtime identity, freshness validation, and protocol repair."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .permissions import PermissionPolicy
from .state import SCHEMA_VERSION, Checkpoint, SessionState
from .tools.base import Tool
from .workspace import WorkspaceContext

RecoveryStatus = Literal[
    "full-valid",
    "files-stale",
    "runtime-mismatch",
    "incomplete-tool-call",
    "schema-mismatch",
]

INTERRUPTED_TOOL_RESULT = (
    "[interrupted: previous execution state is unknown; "
    "inspect workspace before retrying]"
)
UNVERIFIABLE_FRESHNESS = "unverifiable-directory-search"
_RUNTIME_KEYS = (
    "model",
    "repo_root",
    "branch",
    "tool_signature",
    "permission_mode",
    "harness_schema_version",
)


@dataclass(frozen=True)
class RecoveryResult:
    status: RecoveryStatus
    checkpoint_id: str | None = None
    stale_paths: list[str] = field(default_factory=list)
    runtime_differences: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_tool_names: list[str] = field(default_factory=list)
    requires_workspace_inspection: bool = False
    notice: str | None = None

    @property
    def can_resume(self) -> bool:
        return self.status != "schema-mismatch"


def build_runtime_identity(
    *,
    model: str,
    workspace: WorkspaceContext,
    tools: list[Tool],
    permission_policy: PermissionPolicy,
) -> dict[str, Any]:
    """Build the non-secret identity that makes a checkpoint reproducible."""
    tool_payload = [
        {
            "name": tool.name,
            "read_only": tool.read_only,
            "risk_level": tool.risk_level,
            "schema": tool.schema(),
        }
        for tool in sorted(tools, key=lambda item: item.name)
    ]
    serialized_tools = json.dumps(
        tool_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "model": model,
        "repo_root": str(workspace.repo_root),
        "branch": workspace.current_branch(),
        "tool_signature": hashlib.sha256(serialized_tools.encode("utf-8")).hexdigest(),
        "permission_mode": permission_policy.mode,
        "harness_schema_version": SCHEMA_VERSION,
    }


def evaluate_recovery(
    session: SessionState,
    checkpoint: Checkpoint | None,
    *,
    current_runtime: dict[str, Any],
    workspace: WorkspaceContext,
) -> RecoveryResult:
    """Classify recovery without mutating messages or executing tools."""
    pending = find_pending_tool_calls(session.messages)
    answered_ids = {
        message.get("tool_call_id")
        for message in session.messages
        if message.get("role") == "tool"
    }
    pending_ids = {call["id"] for call in pending}
    runtime_differences: dict[str, dict[str, Any]] = {}
    stale_paths: list[str] = []

    if checkpoint is None:
        runtime_differences["checkpoint"] = {
            "expected": "present",
            "actual": "missing",
        }
    else:
        if checkpoint.session_id != session.session_id:
            runtime_differences["session_id"] = {
                "expected": checkpoint.session_id,
                "actual": session.session_id,
            }
        for key in _RUNTIME_KEYS:
            expected = checkpoint.runtime_identity.get(key)
            actual = current_runtime.get(key)
            if expected != actual:
                runtime_differences[key] = {"expected": expected, "actual": actual}
        for path, expected_hash in sorted(checkpoint.file_freshness.items()):
            if expected_hash == UNVERIFIABLE_FRESHNESS:
                stale_paths.append(path)
                continue
            try:
                _, actual_hash = workspace.fingerprint_path(path)
            except (OSError, ValueError):
                actual_hash = "unavailable"
            if expected_hash != actual_hash:
                stale_paths.append(path)
        for checkpoint_call in checkpoint.pending_tool_calls:
            call = _normalize_checkpoint_tool_call(checkpoint_call)
            if call["id"] not in answered_ids and call["id"] not in pending_ids:
                pending.append(call)
                pending_ids.add(call["id"])

    if pending:
        status: RecoveryStatus = "incomplete-tool-call"
    elif runtime_differences:
        status = "runtime-mismatch"
    elif stale_paths:
        status = "files-stale"
    else:
        status = "full-valid"

    pending_names = [call["name"] for call in pending]
    notice = _build_notice(status, stale_paths, runtime_differences, pending_names)
    return RecoveryResult(
        status=status,
        checkpoint_id=checkpoint.checkpoint_id if checkpoint is not None else None,
        stale_paths=stale_paths,
        runtime_differences=runtime_differences,
        pending_tool_names=pending_names,
        requires_workspace_inspection=bool(pending),
        notice=notice,
    )


def schema_mismatch_result(checkpoint_id: str | None) -> RecoveryResult:
    return RecoveryResult(
        status="schema-mismatch",
        checkpoint_id=checkpoint_id,
        notice=(
            "[PikaCore recovery: schema-mismatch]\n"
            "The saved checkpoint uses an unsupported schema and was left unchanged."
        ),
    )


def apply_recovery(session: SessionState, result: RecoveryResult) -> bool:
    """Repair incomplete calls and append deterministic recovery context."""
    if not result.can_resume:
        return False
    repaired_messages, changed = repair_pending_tool_calls(session.messages)
    if changed:
        session.messages = repaired_messages
    if result.notice is not None and not _notice_already_present(
        session.messages, result.notice
    ):
        session.messages.append({"role": "user", "content": result.notice})
        changed = True
    if changed:
        session.touch()
    return changed


def find_pending_tool_calls(messages: list[dict]) -> list[dict[str, Any]]:
    """Return assistant calls without a matching result, in model order."""
    answered = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    pending = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw_call in message.get("tool_calls") or []:
            call = _normalize_message_tool_call(raw_call)
            if call["id"] not in answered:
                pending.append(call)
    return pending


def repair_pending_tool_calls(messages: list[dict]) -> tuple[list[dict], bool]:
    """Insert one interrupted result per orphan call without invoking a tool."""
    repaired = copy.deepcopy(messages)
    pending = find_pending_tool_calls(repaired)
    if not pending:
        return repaired, False

    pending_by_id = {call["id"]: call for call in pending}
    output: list[dict] = []
    index = 0
    while index < len(repaired):
        message = repaired[index]
        output.append(message)
        index += 1
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue

        existing_results = []
        while index < len(repaired) and repaired[index].get("role") == "tool":
            existing_results.append(repaired[index])
            index += 1
        existing_by_id = {
            result.get("tool_call_id"): result for result in existing_results
        }
        expected_ids = []
        for raw_call in message["tool_calls"]:
            call = _normalize_message_tool_call(raw_call)
            expected_ids.append(call["id"])
            if call["id"] in existing_by_id:
                output.append(existing_by_id[call["id"]])
            elif call["id"] in pending_by_id:
                output.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": INTERRUPTED_TOOL_RESULT,
                })
        output.extend(
            result
            for result in existing_results
            if result.get("tool_call_id") not in expected_ids
        )
    return output, True


def serialize_tool_call(tool_call) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": copy.deepcopy(tool_call.arguments),
    }


def _normalize_message_tool_call(raw_call: dict) -> dict[str, Any]:
    function = raw_call.get("function") or {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    return {
        "id": raw_call.get("id"),
        "name": function.get("name", raw_call.get("name", "unknown")),
        "arguments": arguments,
    }


def _normalize_checkpoint_tool_call(raw_call: dict) -> dict[str, Any]:
    return {
        "id": raw_call.get("id"),
        "name": raw_call.get("name", "unknown"),
        "arguments": copy.deepcopy(raw_call.get("arguments", {})),
    }


def _build_notice(
    status: RecoveryStatus,
    stale_paths: list[str],
    runtime_differences: dict[str, dict[str, Any]],
    pending_tool_names: list[str],
) -> str | None:
    if status == "full-valid":
        return None
    lines = [f"[PikaCore recovery: {status}]"]
    if pending_tool_names:
        lines.append("Pending tools were not replayed: " + ", ".join(pending_tool_names))
        lines.append(
            "Inspect workspace changes before deciding whether any tool should be retried."
        )
    if stale_paths:
        lines.append("Re-read stale paths before relying on saved context: " + ", ".join(stale_paths))
    if runtime_differences:
        lines.append(
            "Runtime differences require review: "
            + ", ".join(sorted(runtime_differences))
        )
    return "\n".join(lines)


def _notice_already_present(messages: list[dict], notice: str) -> bool:
    return bool(
        messages
        and messages[-1].get("role") == "user"
        and messages[-1].get("content") == notice
    )
