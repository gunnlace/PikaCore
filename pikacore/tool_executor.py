"""Structured tool execution with permissions and read/write barriers."""

import concurrent.futures
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .permissions import PermissionPolicy
from .tools.base import Tool
from .workspace import WorkspaceSnapshot

ApprovalCallback = Callable[[Tool, dict], bool]


@dataclass
class ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    content: str
    status: Literal["ok", "error", "rejected", "partial"]
    error_code: str | None = None
    duration_ms: int = 0
    read_paths: list[str] = field(default_factory=list)
    freshness_review_paths: list[str] = field(default_factory=list)
    affected_paths: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    exit_code: int | None = None
    output_truncated: bool = False
    approval: Literal["not_required", "approved", "rejected"] = "not_required"


class ToolExecutor:
    def __init__(
        self,
        tools: list[Tool],
        *,
        permission_policy: PermissionPolicy | None = None,
        approval_callback: ApprovalCallback | None = None,
        max_workers: int = 8,
    ):
        self.tools = tools
        self._tool_by_name = {tool.name: tool for tool in tools}
        self.permission_policy = permission_policy or PermissionPolicy()
        self.approval_callback = approval_callback
        self.max_workers = max_workers

    def execute_one(self, tool_call) -> ToolExecutionResult:
        started = time.perf_counter()
        tool = self._tool_by_name.get(tool_call.name)
        if tool is None:
            return self._result(
                tool_call,
                content=f"Error: unknown tool '{tool_call.name}'",
                status="error",
                error_code="unknown-tool",
                started=started,
            )

        signature = inspect.signature(tool.execute)
        try:
            bound_arguments = signature.bind(**tool_call.arguments)
            bound_arguments.apply_defaults()
        except TypeError as exc:
            return self._result(
                tool_call,
                content=f"Error: bad arguments for {tool_call.name}: {exc}",
                status="error",
                error_code="bad-arguments",
                started=started,
            )

        try:
            arguments = tool.normalize_arguments(
                self._expanded_arguments(signature, bound_arguments)
            )
        except (OSError, ValueError) as exc:
            return self._result(
                tool_call,
                content=f"Error: {exc}",
                status="error",
                error_code="path-rejected",
                started=started,
            )

        decision = self.permission_policy.decide(tool)
        if decision == "deny":
            return self._rejected(tool_call, started, "permission-denied")
        approval = "not_required"
        if decision == "ask":
            approved = self.approval_callback(tool, dict(arguments)) if self.approval_callback else False
            if not approved:
                return self._rejected(tool_call, started, "approval-rejected")
            approval = "approved"

        before = tool.workspace_context.snapshot() if not tool.read_only else None
        try:
            output = tool.execute_structured(**arguments)
        except Exception as exc:
            detected_paths, workspace_changed = self._workspace_delta(tool, before)
            return self._result(
                tool_call,
                content=f"Error executing {tool_call.name}: {exc}",
                status="error",
                error_code="tool-error",
                started=started,
                approval=approval,
                arguments=arguments,
                tool=tool,
                detected_paths=detected_paths,
                workspace_changed=workspace_changed,
            )

        detected_paths, workspace_changed = self._workspace_delta(tool, before)
        return self._result(
            tool_call,
            content=output.content,
            status=output.status,
            error_code=output.error_code,
            started=started,
            approval=approval,
            arguments=arguments,
            tool=tool,
            detected_paths=detected_paths,
            workspace_changed=workspace_changed,
            exit_code=output.exit_code,
            output_truncated=output.output_truncated,
            reported_read_paths=(
                list(output.read_paths) if output.read_paths is not None else None
            ),
            freshness_review_paths=list(output.freshness_review_paths),
        )

    @staticmethod
    def _expanded_arguments(
        signature: inspect.Signature,
        bound_arguments: inspect.BoundArguments,
    ) -> dict:
        """Return named arguments while expanding an execute(**kwargs) bucket."""
        arguments = {}
        for name, value in bound_arguments.arguments.items():
            if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                arguments.update(value)
            else:
                arguments[name] = value
        return arguments

    @staticmethod
    def _workspace_delta(
        tool: Tool,
        before: WorkspaceSnapshot | None,
    ) -> tuple[list[str], bool]:
        if tool.read_only:
            return [], False
        if before is None:
            # Non-Git workspaces intentionally avoid a full directory traversal.
            # Once a mutating tool starts, any outcome may include side effects.
            return [], True
        after = tool.workspace_context.snapshot()
        if after is None:
            return [], True
        before_paths = dict(before.path_fingerprints)
        after_paths = dict(after.path_fingerprints)
        relative_paths = sorted(
            path
            for path in before_paths.keys() | after_paths.keys()
            if before_paths.get(path) != after_paths.get(path)
        )
        if not relative_paths:
            return [], False
        detected_paths = []
        for relative_path in relative_paths:
            try:
                detected_paths.append(
                    str(tool.workspace_context.resolve_path(relative_path, for_write=True))
                )
            except ValueError:
                continue
        return detected_paths, True

    def execute_many(
        self,
        tool_calls,
        on_tool=None,
        on_result=None,
    ) -> list[ToolExecutionResult]:
        """Parallelize consecutive reads and serialize every mutating barrier."""
        for tool_call in tool_calls:
            if on_tool:
                on_tool(tool_call.name, tool_call.arguments)

        results: list[ToolExecutionResult | None] = [None] * len(tool_calls)
        index = 0
        reject_following_barrier = False
        while index < len(tool_calls):
            tool = self._tool_by_name.get(tool_calls[index].name)
            if tool is not None and tool.read_only:
                end = index
                while end < len(tool_calls):
                    candidate = self._tool_by_name.get(tool_calls[end].name)
                    if candidate is None or not candidate.read_only:
                        break
                    end += 1
                self._execute_read_batch(tool_calls, results, index, end)
                if on_result:
                    for result_index in range(index, end):
                        on_result(tool_calls[result_index], results[result_index])
                reject_following_barrier = False
                index = end
                continue

            if reject_following_barrier:
                results[index] = self._rejected(
                    tool_calls[index],
                    time.perf_counter(),
                    "rejected-by-barrier",
                )
            else:
                result = self.execute_one(tool_calls[index])
                results[index] = result
                reject_following_barrier = result.status == "rejected"
            if on_result:
                on_result(tool_calls[index], results[index])
            index += 1

        return [result for result in results if result is not None]

    def _execute_read_batch(self, tool_calls, results, start: int, end: int):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, end - start)
        ) as pool:
            futures = [pool.submit(self.execute_one, tool_calls[index]) for index in range(start, end)]
            for offset, future in enumerate(futures):
                results[start + offset] = future.result()

    def _rejected(self, tool_call, started: float, error_code: str) -> ToolExecutionResult:
        return self._result(
            tool_call,
            content=f"Error: permission rejected for {tool_call.name}",
            status="rejected",
            error_code=error_code,
            started=started,
            approval="rejected",
        )

    @staticmethod
    def _result(
        tool_call,
        *,
        content: str,
        status: Literal["ok", "error", "rejected", "partial"],
        error_code: str | None,
        started: float,
        approval: Literal["not_required", "approved", "rejected"] = "not_required",
        arguments: dict | None = None,
        tool: Tool | None = None,
        detected_paths: list[str] | None = None,
        workspace_changed: bool = False,
        exit_code: int | None = None,
        output_truncated: bool = False,
        reported_read_paths: list[str] | None = None,
        freshness_review_paths: list[str] | None = None,
    ) -> ToolExecutionResult:
        arguments = arguments or {}
        path = arguments.get("file_path") or arguments.get("path")
        read_paths = (
            reported_read_paths
            if reported_read_paths is not None
            else [path] if path and tool and tool.read_only else []
        )
        explicit_paths = [path] if path and tool and not tool.read_only and status == "ok" else []
        affected_paths = list(dict.fromkeys(explicit_paths + (detected_paths or [])))
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=content,
            status=status,
            error_code=error_code,
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
            read_paths=read_paths,
            freshness_review_paths=freshness_review_paths or [],
            affected_paths=affected_paths,
            workspace_changed=bool(explicit_paths) or workspace_changed,
            exit_code=exit_code,
            output_truncated=output_truncated,
            approval=approval,
        )
