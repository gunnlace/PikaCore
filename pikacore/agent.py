"""Core agent loop.

This is the heart of PikaCore.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import time
import warnings
from collections import Counter
from dataclasses import dataclass, field

from .checkpoint import (
    INTERRUPTED_TOOL_RESULT,
    RecoveryResult,
    UNVERIFIABLE_FRESHNESS,
    apply_recovery,
    build_runtime_identity,
    evaluate_recovery,
    schema_mismatch_result,
    serialize_tool_call,
)
from .llm import LLM, ToolCall
from .tools import create_tools
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager, estimate_tokens
from .permissions import PermissionPolicy
from .state import (
    Checkpoint,
    Report,
    RunState,
    SchemaMismatchError,
    SessionState,
    TraceEvent,
    WorkingMemory,
    utc_now,
)
from .store import ProjectStore
from .tool_executor import ApprovalCallback, ToolExecutionResult, ToolExecutor
from .workspace import WorkspaceContext
from .working_memory import (
    CheckpointMemoryEvent,
    RecoveryMemoryEvent,
    RunMemoryEvent,
    ToolMemoryEvent,
    UserMemoryEvent,
    WorkingMemoryManager,
    render_working_memory,
)


@dataclass
class _RunMetrics:
    started_perf: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: Counter = field(default_factory=Counter)
    tool_errors: Counter = field(default_factory=Counter)
    tool_approvals: Counter = field(default_factory=Counter)
    tool_error_count: int = 0
    approval_count: int = 0
    read_paths: set[str] = field(default_factory=set)
    affected_paths: set[str] = field(default_factory=set)
    context_compressions: int = 0
    context_tokens_before: int = 0
    context_tokens_after: int = 0
    context_compression_events: list[dict] = field(default_factory=list)
    persistence_errors: list[str] = field(default_factory=list)


class CheckpointPersistenceError(RuntimeError):
    """Raised when safe continuation requires a checkpoint that cannot be saved."""


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        workspace: WorkspaceContext | None = None,
        permission_policy: PermissionPolicy | None = None,
        approval_callback: ApprovalCallback | None = None,
        store: ProjectStore | None = None,
        session_state: SessionState | None = None,
        persist: bool = True,
        parent_run_id: str | None = None,
    ):
        self.llm = llm
        self.workspace = workspace or WorkspaceContext.discover()
        self.tools = tools if tools is not None else create_tools(self.workspace)
        for tool in self.tools:
            tool.bind_workspace(self.workspace)
        self._tool_by_name = {t.name: t for t in self.tools}
        self.permission_policy = permission_policy or PermissionPolicy()
        self.approval_callback = approval_callback
        self.tool_executor = ToolExecutor(
            self.tools,
            permission_policy=self.permission_policy,
            approval_callback=self.approval_callback,
        )
        model = getattr(self.llm, "model", "unknown")
        self.session_state = session_state or SessionState(
            repo_root=str(self.workspace.repo_root),
            model=model,
        )
        self.session_state.working_memory = WorkingMemory.from_dict(
            self.session_state.working_memory
        )
        self.working_memory = WorkingMemoryManager(
            self.session_state.working_memory,
            self.workspace,
        )
        self.store = store if store is not None else (
            ProjectStore(repo_root=self.workspace.repo_root) if persist else None
        )
        self.parent_run_id = parent_run_id
        self.current_run: RunState | None = None
        self._run_metrics: _RunMetrics | None = None
        self._trace_seq = 0
        self._completed_tool_call_ids: list[str] = []
        self._pending_tool_calls: list[dict] = []
        self._file_freshness: dict[str, str] = {}
        self._last_successful_action: str | None = None
        self._checkpoint_status: str | None = None
        self.recovery_result: RecoveryResult | None = None
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        full = [{"role": "system", "content": self._system}]
        rendered_memory = render_working_memory(self.session_state.working_memory)
        if rendered_memory:
            full.append({"role": "system", "content": rendered_memory})
        return full + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    @property
    def messages(self) -> list[dict]:
        return self.session_state.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self.session_state.messages = value

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        run = self._start_run(user_input)
        try:
            self._append_message({"role": "user", "content": user_input})
            self._maybe_compress()

            for _ in range(self.max_rounds):
                self._trace(
                    "context_built",
                    {
                        "message_count": len(self.messages),
                        "estimated_tokens": estimate_tokens(self._full_messages()),
                    },
                )
                run.model_attempts += 1
                self._save_run()
                self._trace("model_requested", {"attempt": run.model_attempts})
                try:
                    resp = self.llm.chat(
                        messages=self._full_messages(),
                        tools=self._tool_schemas(),
                        on_token=on_token,
                    )
                except Exception:
                    run.stop_reason = "model_error"
                    raise

                metrics = self._run_metrics
                if metrics is not None:
                    metrics.prompt_tokens += resp.prompt_tokens
                    metrics.completion_tokens += resp.completion_tokens
                self._trace(
                    "model_completed",
                    {
                        "attempt": run.model_attempts,
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                        "tool_call_count": len(resp.tool_calls),
                    },
                )

                if not resp.tool_calls:
                    self._append_message(resp.message)
                    self._finish_run(
                        status="completed",
                        stop_reason="completed",
                        final_answer=resp.content,
                    )
                    return resp.content

                # Persist assistant tool calls before any side effect begins.
                self._append_message(resp.message)
                for tool_call in resp.tool_calls:
                    self._trace(
                        "tool_requested",
                        {
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    )
                self._pending_tool_calls = [
                    serialize_tool_call(tool_call) for tool_call in resp.tool_calls
                ]
                self._require_checkpoint(
                    last_successful_action="assistant_tool_calls_saved",
                    next_suggested_action=(
                        "inspect workspace before deciding whether to retry pending tools"
                    ),
                )

                try:
                    result_index = 0

                    def persist_result(tool_call, result):
                        nonlocal result_index
                        self._record_tool_result(tool_call, result)
                        self._append_message({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.content,
                        })
                        tool = self._tool_by_name.get(tool_call.name)
                        next_tool = (
                            self._tool_by_name.get(
                                resp.tool_calls[result_index + 1].name
                            )
                            if result_index + 1 < len(resp.tool_calls)
                            else None
                        )
                        read_batch_finished = bool(
                            tool is not None
                            and tool.read_only
                            and (next_tool is None or not next_tool.read_only)
                        )
                        if tool is None or not tool.read_only or read_batch_finished:
                            self._require_checkpoint()
                        result_index += 1

                    self.tool_executor.execute_many(
                        resp.tool_calls,
                        on_tool=on_tool,
                        on_result=persist_result,
                    )
                except KeyboardInterrupt:
                    # Persist one backfill per pending call before propagating.
                    self._answer_pending_tool_calls(resp.tool_calls)
                    raise
                except Exception:
                    # Executor/callback failures must not persist orphan calls.
                    self._answer_pending_tool_calls(resp.tool_calls)
                    raise

                self._maybe_compress()

            answer = "(reached maximum tool-call rounds)"
            self._finish_run(
                status="failed",
                stop_reason="max_rounds",
                final_answer=answer,
            )
            return answer
        except KeyboardInterrupt:
            if self.current_run is run:
                self._finish_run(
                    status="interrupted",
                    stop_reason="user_interrupted",
                    error="KeyboardInterrupt",
                )
            raise
        except Exception as exc:
            if self.current_run is run:
                self._finish_run(
                    status="failed",
                    stop_reason=run.stop_reason or "internal_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise

    def _exec_tool(self, tc) -> str:
        """Compatibility facade returning only model-visible result content."""
        return self.tool_executor.execute_one(tc).content

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Compatibility facade preserving the historical list[str] result."""
        return [
            result.content
            for result in self.tool_executor.execute_many(tool_calls, on_tool=on_tool)
        ]

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self._append_message({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": INTERRUPTED_TOOL_RESULT,
                })
                self._pending_tool_calls = [
                    call for call in self._pending_tool_calls if call.get("id") != tc.id
                ]

    def _start_run(self, user_input: str) -> RunState:
        if self.current_run is not None:
            raise RuntimeError("Agent already has an active run")
        run = RunState(
            session_id=self.session_state.session_id,
            user_request=user_input,
        )
        self.current_run = run
        self._trace_seq = 0
        self._completed_tool_call_ids = []
        self._pending_tool_calls = []
        self._last_successful_action = None
        self._checkpoint_status = None
        self._run_metrics = _RunMetrics(
            started_perf=time.perf_counter(),
        )
        self.session_state.run_ids.append(run.run_id)
        self.session_state.touch()
        memory_changed = self.working_memory.apply(UserMemoryEvent(
            request=user_input,
            run_id=run.run_id,
            occurred_at=utc_now(),
        ))
        self._save_run()
        self._save_session()
        self._trace(
            "run_started",
            {
                "model": getattr(self.llm, "model", "unknown"),
                "repo_root": str(self.workspace.repo_root),
                "parent_run_id": self.parent_run_id,
            },
        )
        if memory_changed:
            self._trace_working_memory("user")
        return run

    def _append_message(self, message: dict) -> None:
        self.messages.append(message)
        self._save_session()
        self._trace(
            "message_appended",
            {
                "role": message.get("role"),
                "tool_call_id": message.get("tool_call_id"),
                "has_tool_calls": bool(message.get("tool_calls")),
            },
        )

    def _maybe_compress(self) -> bool:
        result = self.context.maybe_compress(
            self.messages,
            self.llm,
            self.session_state.working_memory,
        )
        if not result.changed:
            return False
        metrics = self._run_metrics
        event = {
            "strategy": result.strategy,
            "before_tokens": result.before_tokens,
            "after_tokens": result.after_tokens,
            "removed_messages": result.removed_messages,
            "summarized_messages": result.summarized_messages,
        }
        if metrics is not None:
            metrics.context_compressions += 1
            if metrics.context_compressions == 1:
                metrics.context_tokens_before = result.before_tokens
            metrics.context_tokens_after = result.after_tokens
            metrics.context_compression_events.append(event)
        self._save_session()
        self._trace("context_compressed", event)
        self._require_checkpoint(last_successful_action="context_compressed")
        return True

    def _record_tool_result(self, tool_call, result) -> None:
        run = self.current_run
        metrics = self._run_metrics
        if run is None or metrics is None:
            return
        run.tool_steps += 1
        metrics.tool_calls[tool_call.name] += 1
        metrics.tool_error_count += result.status != "ok"
        metrics.approval_count += result.approval in {"approved", "rejected"}
        if result.status != "ok":
            metrics.tool_errors[tool_call.name] += 1
        if result.approval in {"approved", "rejected"}:
            metrics.tool_approvals[tool_call.name] += 1
        metrics.read_paths.update(result.read_paths)
        metrics.affected_paths.update(result.affected_paths)
        if tool_call.id not in self._completed_tool_call_ids:
            self._completed_tool_call_ids.append(tool_call.id)
        self._pending_tool_calls = [
            call for call in self._pending_tool_calls if call.get("id") != tool_call.id
        ]
        self._refresh_file_freshness(result.read_paths + result.affected_paths)
        self._mark_freshness_review(result.freshness_review_paths)
        if result.status == "ok":
            self._last_successful_action = f"tool:{tool_call.name}"
        self._save_run()

        event_data = {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "status": result.status,
            "error_code": result.error_code,
            "duration_ms": result.duration_ms,
            "read_paths": result.read_paths,
            "freshness_review_paths": result.freshness_review_paths,
            "affected_paths": result.affected_paths,
            "workspace_changed": result.workspace_changed,
            "exit_code": result.exit_code,
            "output_truncated": result.output_truncated,
            "approval": result.approval,
            "output_summary": result.content[:1000],
        }
        if result.approval == "approved":
            self._trace("tool_approved", event_data)
        if result.status == "rejected":
            self._trace("tool_rejected", event_data)
        self._trace("tool_completed", event_data)
        if self.working_memory.apply(ToolMemoryEvent(
            tool_name=tool_call.name,
            arguments=dict(tool_call.arguments),
            result=result,
            run_id=run.run_id,
            occurred_at=utc_now(),
        )):
            self._trace_working_memory("tool")

    def _finish_run(
        self,
        *,
        status: str,
        stop_reason: str,
        final_answer: str | None = None,
        error: str | None = None,
    ) -> None:
        run = self.current_run
        metrics = self._run_metrics
        if run is None or metrics is None:
            return
        run.status = status
        run.stop_reason = stop_reason
        run.finished_at = utc_now()
        run.final_answer = final_answer
        run.error = error
        self._save_run()
        if status == "completed":
            self._last_successful_action = "run_completed"
        self._create_checkpoint(last_successful_action=self._last_successful_action)
        if self.working_memory.apply(RunMemoryEvent(
            status=status,
            stop_reason=stop_reason,
            run_id=run.run_id,
            occurred_at=run.finished_at,
        )):
            self._trace_working_memory("run")
        self._trace(
            "run_finished" if status == "completed" else "run_failed",
            {"status": status, "stop_reason": stop_reason, "error": error},
        )
        # Complete the Session durability point before freezing report metrics so
        # a recoverable final-session failure is represented in the report.
        self._save_session()

        report = Report(
            run_id=run.run_id,
            session_id=run.session_id,
            model=getattr(self.llm, "model", "unknown"),
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_ms=max(0, int((time.perf_counter() - metrics.started_perf) * 1000)),
            model_attempts=run.model_attempts,
            tool_steps=run.tool_steps,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            tool_calls=dict(sorted(metrics.tool_calls.items())),
            tool_errors=dict(sorted(metrics.tool_errors.items())),
            tool_approvals=dict(sorted(metrics.tool_approvals.items())),
            tool_error_count=metrics.tool_error_count,
            approval_count=metrics.approval_count,
            read_paths=sorted(metrics.read_paths),
            affected_paths=sorted(metrics.affected_paths),
            context_compressions=metrics.context_compressions,
            context_tokens_before=metrics.context_tokens_before,
            context_tokens_after=metrics.context_tokens_after,
            context_compression_events=list(metrics.context_compression_events),
            checkpoint_status=self._checkpoint_status,
            recovery_status=(
                self.recovery_result.status if self.recovery_result is not None else None
            ),
            stop_reason=stop_reason,
            completed=status == "completed",
            error=error,
            persistence_errors=list(metrics.persistence_errors),
        )
        if self.store is not None:
            self._safe_store("save report", self.store.save_report, report)
        self.current_run = None
        self._run_metrics = None

    def recover_session(self) -> RecoveryResult:
        """Validate and safely repair a loaded session without replaying tools."""
        checkpoint_id = self.session_state.last_checkpoint_id
        checkpoint = None
        if checkpoint_id is not None and self.store is not None:
            try:
                checkpoint = self.store.load_checkpoint(checkpoint_id)
            except (SchemaMismatchError, TypeError, ValueError, KeyError, OSError):
                result = schema_mismatch_result(checkpoint_id)
                self.recovery_result = result
                return result

        if checkpoint is not None and checkpoint.session_id == self.session_state.session_id:
            self._file_freshness = dict(checkpoint.file_freshness)

        result = evaluate_recovery(
            self.session_state,
            checkpoint,
            current_runtime=self._runtime_identity(),
            workspace=self.workspace,
        )
        self.recovery_result = result
        session_changed = apply_recovery(self.session_state, result)
        memory_changed = self.working_memory.apply(RecoveryMemoryEvent(
            result=result,
            occurred_at=utc_now(),
        ))
        if session_changed or memory_changed:
            self._save_session()
        return result

    def _runtime_identity(self) -> dict:
        return build_runtime_identity(
            model=getattr(self.llm, "model", "unknown"),
            workspace=self.workspace,
            tools=self.tools,
            permission_policy=self.permission_policy,
        )

    def _refresh_file_freshness(self, paths: list[str]) -> None:
        for path in paths:
            try:
                relative, fingerprint = self.workspace.fingerprint_path(path)
            except (OSError, ValueError):
                continue
            self._file_freshness[relative] = fingerprint

    def _mark_freshness_review(self, paths: list[str]) -> None:
        for path in paths:
            try:
                resolved = self.workspace.resolve_path(path)
                relative = resolved.relative_to(self.workspace.repo_root).as_posix()
            except (OSError, ValueError):
                continue
            self._file_freshness[relative] = UNVERIFIABLE_FRESHNESS

    def _require_checkpoint(
        self,
        *,
        last_successful_action: str | None = None,
        next_suggested_action: str | None = None,
    ) -> Checkpoint | None:
        checkpoint = self._create_checkpoint(
            last_successful_action=last_successful_action,
            next_suggested_action=next_suggested_action,
        )
        if self.store is not None and checkpoint is None:
            self._fail_pending_for_checkpoint_error()
            raise CheckpointPersistenceError(
                "required checkpoint could not be persisted; pending tools were not executed"
            )
        return checkpoint

    def _fail_pending_for_checkpoint_error(self) -> None:
        pending = list(self._pending_tool_calls)
        for call in pending:
            tool_call = ToolCall(
                id=call.get("id"),
                name=call.get("name", "unknown"),
                arguments=call.get("arguments", {}),
            )
            result = ToolExecutionResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=(
                    "Error: tool was not executed because the required checkpoint "
                    "could not be persisted."
                ),
                status="error",
                error_code="checkpoint-failed",
            )
            self._record_tool_result(tool_call, result)
            self._append_message({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.content,
            })

    def _create_checkpoint(
        self,
        *,
        last_successful_action: str | None = None,
        next_suggested_action: str | None = None,
    ) -> Checkpoint | None:
        if self.store is None or self.current_run is None:
            return None
        if last_successful_action is not None:
            self._last_successful_action = last_successful_action
        resolved_next_action = (
            next_suggested_action
            if next_suggested_action is not None
            else (
                "inspect workspace before deciding whether to retry pending tools"
                if self._pending_tool_calls
                else None
            )
        )
        checkpoint = Checkpoint(
            parent_checkpoint_id=self.session_state.last_checkpoint_id,
            session_id=self.session_state.session_id,
            run_id=self.current_run.run_id,
            current_user_request=self.current_run.user_request,
            completed_tool_call_ids=list(self._completed_tool_call_ids),
            pending_tool_calls=list(self._pending_tool_calls),
            last_successful_action=self._last_successful_action,
            next_suggested_action=resolved_next_action,
            file_freshness=dict(sorted(self._file_freshness.items())),
            runtime_identity=self._runtime_identity(),
        )
        if not self._safe_store(
            "save checkpoint", self.store.save_checkpoint, checkpoint
        ):
            self._checkpoint_status = "error"
            return None
        self.session_state.last_checkpoint_id = checkpoint.checkpoint_id
        if self._checkpoint_status != "error":
            self._checkpoint_status = "created"
        if self.working_memory.apply(CheckpointMemoryEvent(
            file_freshness=checkpoint.file_freshness,
            pending_tool_names=[
                str(call.get("name", "unknown"))
                for call in checkpoint.pending_tool_calls
            ],
            next_suggested_action=checkpoint.next_suggested_action,
            occurred_at=checkpoint.created_at,
        )):
            self._trace_working_memory("checkpoint")
        self._save_session()
        self._trace(
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "pending_tool_count": len(checkpoint.pending_tool_calls),
                "completed_tool_count": len(checkpoint.completed_tool_call_ids),
            },
        )
        return checkpoint

    def _save_session(self) -> None:
        self.session_state.model = getattr(self.llm, "model", "unknown")
        self.session_state.touch()
        if self.store is not None:
            self._safe_store("save session", self.store.save_session, self.session_state)

    def _save_run(self) -> None:
        if self.store is not None and self.current_run is not None:
            self._safe_store("save run", self.store.save_run, self.current_run)

    def _trace(self, event: str, data: dict) -> None:
        if self.store is None or self.current_run is None:
            return
        self._trace_seq += 1
        trace_event = TraceEvent(
            seq=self._trace_seq,
            event=event,
            session_id=self.session_state.session_id,
            run_id=self.current_run.run_id,
            data=data,
        )
        self._safe_store("append trace", self.store.append_trace, trace_event)

    def _trace_working_memory(self, source: str) -> None:
        memory = self.session_state.working_memory
        self._trace(
            "working_memory_updated",
            {
                "source": source,
                "file_count": len(memory.files),
                "command_count": len(memory.recent_commands),
                "blocker_count": len(memory.blockers),
                "next_step_count": len(memory.next_steps),
            },
        )

    def _safe_store(self, action: str, operation, *args) -> bool:
        try:
            operation(*args)
            return True
        except Exception as exc:
            message = f"{action} failed: {type(exc).__name__}: {exc}"
            if self._run_metrics is not None:
                self._run_metrics.persistence_errors.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return False

    def reset(self):
        """Clear conversation, Working Memory, and recovery continuity."""
        self.messages.clear()
        self.session_state.working_memory = WorkingMemory()
        self.working_memory = WorkingMemoryManager(
            self.session_state.working_memory,
            self.workspace,
        )
        self.session_state.last_checkpoint_id = None
        self._file_freshness.clear()
        self._completed_tool_call_ids.clear()
        self._pending_tool_calls.clear()
        self._last_successful_action = None
        self.recovery_result = None
        self._save_session()
