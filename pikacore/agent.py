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

from .llm import LLM
from .tools import create_tools
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager, estimate_tokens
from .permissions import PermissionPolicy
from .state import Report, RunState, SessionState, TraceEvent, utc_now
from .store import ProjectStore
from .tool_executor import ApprovalCallback, ToolExecutor
from .workspace import WorkspaceContext


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
    persistence_errors: list[str] = field(default_factory=list)


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
        self.store = store if store is not None else (
            ProjectStore(repo_root=self.workspace.repo_root) if persist else None
        )
        self.parent_run_id = parent_run_id
        self.current_run: RunState | None = None
        self._run_metrics: _RunMetrics | None = None
        self._trace_seq = 0
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

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

                try:
                    results = self.tool_executor.execute_many(
                        resp.tool_calls,
                        on_tool=on_tool,
                    )
                    for tool_call, result in zip(resp.tool_calls, results):
                        self._record_tool_result(tool_call, result)
                        self._append_message({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.content,
                        })
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
                    "content": "[interrupted]",
                })

    def _start_run(self, user_input: str) -> RunState:
        if self.current_run is not None:
            raise RuntimeError("Agent already has an active run")
        run = RunState(
            session_id=self.session_state.session_id,
            user_request=user_input,
        )
        self.current_run = run
        self._trace_seq = 0
        self._run_metrics = _RunMetrics(
            started_perf=time.perf_counter(),
        )
        self.session_state.run_ids.append(run.run_id)
        self.session_state.touch()
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
        before = estimate_tokens(self.messages)
        changed = self.context.maybe_compress(self.messages, self.llm)
        if not changed:
            return False
        after = estimate_tokens(self.messages)
        metrics = self._run_metrics
        if metrics is not None:
            metrics.context_compressions += 1
            if metrics.context_compressions == 1:
                metrics.context_tokens_before = before
            metrics.context_tokens_after = after
        self._save_session()
        self._trace(
            "context_compressed",
            {"before_tokens": before, "after_tokens": after},
        )
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
        self._save_run()

        event_data = {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "status": result.status,
            "error_code": result.error_code,
            "duration_ms": result.duration_ms,
            "read_paths": result.read_paths,
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
            stop_reason=stop_reason,
            completed=status == "completed",
            error=error,
            persistence_errors=list(metrics.persistence_errors),
        )
        if self.store is not None:
            self._safe_store("save report", self.store.save_report, report)
        self.current_run = None
        self._run_metrics = None

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

    def _safe_store(self, action: str, operation, *args) -> None:
        try:
            operation(*args)
        except Exception as exc:
            message = f"{action} failed: {type(exc).__name__}: {exc}"
            if self._run_metrics is not None:
                self._run_metrics.persistence_errors.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
        self._save_session()
