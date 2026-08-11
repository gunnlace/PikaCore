"""Offline runner for the ten Phase 6 fixture benchmarks."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pikacore.agent import Agent
from pikacore.context import CompressionResult, estimate_tokens
from pikacore.llm import LLMResponse, ToolCall
from pikacore.permissions import PermissionPolicy
from pikacore.state import Checkpoint, RunState
from pikacore.store import ProjectStore, atomic_write_json
from pikacore.tools import create_tools
from pikacore.tools.base import Tool, ToolOutput
from pikacore.tools.grep import GrepTool
from pikacore.tools.read import ReadFileTool
from pikacore.workspace import WorkspaceContext

from .models import BenchmarkOutcome, BenchmarkSuiteReport
from .scripted_llm import ScriptedFakeLLM

PHASE6_BENCHMARK_IDS = (
    "edit-basic",
    "bad-args-retry",
    "path-escape",
    "permission-reject",
    "parallel-read",
    "write-barrier",
    "large-output",
    "resume-stale-file",
    "resume-unknown-write",
    "working-memory-stale",
)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    working_memory_enabled: bool = True
    context_policy_enabled: bool = True


BASELINE = AblationConfig("baseline")
WORKING_MEMORY_OFF = AblationConfig(
    "working-memory-off",
    working_memory_enabled=False,
)
CONTEXT_POLICY_OFF = AblationConfig(
    "context-policy-off",
    context_policy_enabled=False,
)
DEFAULT_ABLATIONS = (BASELINE, WORKING_MEMORY_OFF, CONTEXT_POLICY_OFF)


@dataclass(frozen=True)
class FixtureSpec:
    benchmark_id: str
    request: str
    responses: tuple[dict[str, Any], ...]
    repo_files: tuple[str, ...]
    permission_mode: str = "auto"
    approvals: tuple[bool, ...] = ()
    max_context_tokens: int = 128_000
    expected: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> FixtureSpec:
        data = json.loads(path.read_text(encoding="utf-8"))
        benchmark_id = str(data["id"])
        if benchmark_id != path.stem:
            raise ValueError(
                f"Fixture id {benchmark_id!r} does not match task file {path.name}"
            )
        return cls(
            benchmark_id=benchmark_id,
            request=str(data.get("request", "")),
            responses=tuple(data.get("responses", [])),
            repo_files=tuple(str(item) for item in data.get("repo_files", [])),
            permission_mode=str(data.get("permission_mode", "auto")),
            approvals=tuple(bool(item) for item in data.get("approvals", [])),
            max_context_tokens=int(data.get("max_context_tokens", 128_000)),
            expected=dict(data.get("expected", {})),
        )


class _DisabledWorkingMemory:
    def apply(self, _event) -> bool:
        return False


class _DisabledContextPolicy:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    @staticmethod
    def maybe_compress(messages, _llm=None, _working_memory=None) -> CompressionResult:
        tokens = estimate_tokens(messages)
        return CompressionResult(False, None, tokens, tokens, 0, 0)


@dataclass
class _ParallelProbe:
    barrier: threading.Barrier = field(default_factory=lambda: threading.Barrier(2))
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread_ids: set[int] = field(default_factory=set)

    def arrive(self) -> None:
        with self.lock:
            self.thread_ids.add(threading.get_ident())
        self.barrier.wait(timeout=3)

    @property
    def ran_in_parallel(self) -> bool:
        return len(self.thread_ids) == 2 and not self.barrier.broken


class _ProbeReadFileTool(ReadFileTool):
    def __init__(self, probe: _ParallelProbe, workspace=None):
        super().__init__(workspace=workspace)
        self.probe = probe

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        self.probe.arrive()
        return super().execute(file_path=file_path, offset=offset, limit=limit)


class _ProbeGrepTool(GrepTool):
    def __init__(self, probe: _ParallelProbe, workspace=None):
        super().__init__(workspace=workspace)
        self.probe = probe

    def execute_structured(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
    ) -> ToolOutput:
        self.probe.arrive()
        return super().execute_structured(pattern=pattern, path=path, include=include)


class _DeterministicLargeBashTool(Tool):
    name = "bash"
    risk_level = "high"
    read_only = False
    description = "Return deterministic large output without spawning a shell."
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def execute(self, command: str) -> str:
        return self.execute_structured(command=command).content

    def execute_structured(self, command: str) -> ToolOutput:
        lines = [f"line-{index:04d}: fixture output" for index in range(900)]
        return ToolOutput(
            content="\n".join(lines),
            status="ok",
            exit_code=0,
            output_truncated=True,
        )


@dataclass
class _Execution:
    agent: Agent
    llm: ScriptedFakeLLM
    store: ProjectStore
    final_answer: str | None = None
    recovery_status: str | None = None
    completed: bool = False
    stop_reason: str | None = None
    exception_type: str | None = None
    extra_checks: dict[str, bool] = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(self, benchmark_root: str | Path | None = None):
        self.root = Path(benchmark_root or Path(__file__).resolve().parent)
        self.tasks_dir = self.root / "tasks"
        self.repos_dir = self.root / "repos"

    def fixture_ids(self) -> tuple[str, ...]:
        ids = tuple(path.stem for path in sorted(self.tasks_dir.glob("*.json")))
        if ids != tuple(sorted(PHASE6_BENCHMARK_IDS)):
            raise ValueError(
                "Phase 6 requires exactly the ten documented fixture benchmark ids"
            )
        return ids

    def load_fixture(self, benchmark_id: str) -> FixtureSpec:
        if benchmark_id not in PHASE6_BENCHMARK_IDS:
            raise ValueError(f"Unknown benchmark fixture: {benchmark_id}")
        spec = FixtureSpec.from_path(self.tasks_dir / f"{benchmark_id}.json")
        if not (self.repos_dir / benchmark_id).is_dir():
            raise ValueError(f"Missing fixture repo for {benchmark_id}")
        if not spec.repo_files:
            raise ValueError(f"Fixture {benchmark_id} has an empty repo_files manifest")
        return spec

    def run_fixture(
        self,
        benchmark_id: str,
        variant: AblationConfig = BASELINE,
    ) -> BenchmarkOutcome:
        spec = self.load_fixture(benchmark_id)
        started = time.perf_counter()
        with TemporaryDirectory(prefix=f"pikacore-{benchmark_id}-") as temporary:
            temporary_root = Path(temporary)
            repo_root = temporary_root / "repo"
            self.materialize_fixture(spec, repo_root)
            state_root = temporary_root / "state"
            execution = self._execute(spec, variant, repo_root, state_root)
            outcome = self._build_outcome(spec, variant, repo_root, execution)
        return BenchmarkOutcome(
            **{
                **outcome.to_dict(),
                "failure_categories": outcome.failure_categories,
                "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
            }
        )

    def materialize_fixture(self, spec: FixtureSpec, destination: Path) -> None:
        """Copy only manifest-listed regular files into an isolated fixture repo."""
        source_root = (self.repos_dir / spec.benchmark_id).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        seen = set()
        for raw_path in spec.repo_files:
            relative = Path(raw_path)
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError(
                    f"Invalid repo_files path for {spec.benchmark_id}: {raw_path}"
                )
            normalized = relative.as_posix()
            if normalized in seen:
                raise ValueError(
                    f"Duplicate repo_files path for {spec.benchmark_id}: {raw_path}"
                )
            seen.add(normalized)
            candidate = source_root / relative
            if candidate.is_symlink():
                raise ValueError(
                    f"repo_files entry is a symlink for "
                    f"{spec.benchmark_id}: {raw_path}"
                )
            source = candidate.resolve()
            try:
                source.relative_to(source_root)
            except ValueError as exc:
                raise ValueError(
                    f"repo_files path escapes fixture {spec.benchmark_id}: {raw_path}"
                ) from exc
            if not source.is_file():
                raise ValueError(
                    f"repo_files entry is not a regular file for "
                    f"{spec.benchmark_id}: {raw_path}"
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

    def run_suite(
        self,
        *,
        fixture_ids: tuple[str, ...] | list[str] | None = None,
        variants: tuple[AblationConfig, ...] | list[AblationConfig] = DEFAULT_ABLATIONS,
    ) -> BenchmarkSuiteReport:
        ids = tuple(fixture_ids or self.fixture_ids())
        configs = tuple(variants)
        started = time.perf_counter()
        outcomes = tuple(
            self.run_fixture(benchmark_id, variant)
            for variant in configs
            for benchmark_id in ids
        )
        count = len(outcomes)
        passed = sum(outcome.passed for outcome in outcomes)
        completed = sum(outcome.completed for outcome in outcomes)
        return BenchmarkSuiteReport(
            schema_version=1,
            suite="pikacore-phase6",
            generated_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
            fixture_count=len(ids),
            variant_count=len(configs),
            outcome_count=count,
            passed_count=passed,
            completed_count=completed,
            success_rate=round(passed / count, 6) if count else 0.0,
            completion_rate=round(completed / count, 6) if count else 0.0,
            outcomes=outcomes,
        )

    @staticmethod
    def write_report(report: BenchmarkSuiteReport, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        atomic_write_json(destination, report.to_dict())
        return destination

    def _execute(
        self,
        spec: FixtureSpec,
        variant: AblationConfig,
        repo_root: Path,
        state_root: Path,
    ) -> _Execution:
        if spec.benchmark_id == "resume-unknown-write":
            return self._execute_unknown_write(spec, variant, repo_root, state_root)

        probe = _ParallelProbe() if spec.benchmark_id == "parallel-read" else None
        agent, llm, store = self._make_agent(
            spec,
            variant,
            repo_root,
            state_root,
            probe=probe,
        )
        execution = _Execution(agent=agent, llm=llm, store=store)
        try:
            execution.final_answer = agent.chat(spec.request)
        except Exception as exc:
            execution.exception_type = type(exc).__name__

        report = self._latest_report(agent, store)
        if report is not None:
            execution.completed = report.completed
            execution.stop_reason = report.stop_reason

        if probe is not None:
            execution.extra_checks["parallel_execution"] = probe.ran_in_parallel

        if spec.benchmark_id == "write-barrier":
            results = {
                message.get("tool_call_id"): str(message.get("content", ""))
                for message in agent.messages
                if message.get("role") == "tool"
            }
            execution.extra_checks["write_barrier_visibility"] = (
                "old" in results.get("read-before", "")
                and "new" in results.get("read-after", "")
            )

        if spec.benchmark_id == "resume-stale-file" and execution.exception_type is None:
            stale_path = repo_root / "state.txt"
            stale_path.write_text("externally changed\n", encoding="utf-8")
            loaded = store.load_session(agent.session_state.session_id)
            if loaded is None:
                execution.extra_checks["session_reload"] = False
            else:
                recovery_agent, _, _ = self._make_agent(
                    spec,
                    variant,
                    repo_root,
                    state_root,
                    responses=(),
                    session_state=loaded,
                )
                recovery = recovery_agent.recover_session()
                execution.recovery_status = recovery.status
                execution.extra_checks["session_reload"] = True
        return execution

    def _execute_unknown_write(
        self,
        spec: FixtureSpec,
        variant: AblationConfig,
        repo_root: Path,
        state_root: Path,
    ) -> _Execution:
        agent, llm, store = self._make_agent(
            spec,
            variant,
            repo_root,
            state_root,
        )
        call = ToolCall(
            "unknown-write",
            "write_file",
            {"file_path": "target.txt", "content": "unexpected\n"},
        )
        agent.messages = [
            {"role": "user", "content": spec.request},
            LLMResponse(tool_calls=[call]).message,
        ]
        run = RunState(
            session_id=agent.session_state.session_id,
            user_request=spec.request,
            status="interrupted",
            stop_reason="user_interrupted",
        )
        checkpoint = Checkpoint(
            session_id=agent.session_state.session_id,
            run_id=run.run_id,
            current_user_request=spec.request,
            pending_tool_calls=[{
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments),
            }],
            next_suggested_action=(
                "inspect workspace before deciding whether to retry pending tools"
            ),
            runtime_identity=agent._runtime_identity(),
        )
        agent.session_state.last_checkpoint_id = checkpoint.checkpoint_id
        store.save_run(run)
        store.save_checkpoint(checkpoint)
        store.save_session(agent.session_state)

        loaded = store.load_session(agent.session_state.session_id)
        if loaded is None:
            return _Execution(
                agent=agent,
                llm=llm,
                store=store,
                exception_type="SessionReloadError",
            )
        recovery_agent, recovery_llm, _ = self._make_agent(
            spec,
            variant,
            repo_root,
            state_root,
            responses=(),
            session_state=loaded,
        )
        recovery = recovery_agent.recover_session()
        target_unchanged = (repo_root / "target.txt").read_text(encoding="utf-8") == "original\n"
        return _Execution(
            agent=recovery_agent,
            llm=recovery_llm,
            store=store,
            recovery_status=recovery.status,
            completed=True,
            extra_checks={
                "no_tool_replay": recovery_llm.remaining == 0 and not recovery_llm.calls,
                "workspace_inspection_required": recovery.requires_workspace_inspection,
                "unknown_write_unchanged": target_unchanged,
            },
        )

    def _make_agent(
        self,
        spec: FixtureSpec,
        variant: AblationConfig,
        repo_root: Path,
        state_root: Path,
        *,
        probe: _ParallelProbe | None = None,
        responses: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
        session_state=None,
    ) -> tuple[Agent, ScriptedFakeLLM, ProjectStore]:
        workspace = WorkspaceContext(repo_root)
        tools = self._tools_for(spec.benchmark_id, workspace, probe)
        llm = ScriptedFakeLLM(list(spec.responses if responses is None else responses))
        approvals = iter(spec.approvals)

        def approve(_tool, _arguments):
            return next(approvals, False)

        store = ProjectStore(state_root=state_root)
        agent = Agent(
            llm=llm,
            tools=tools,
            max_context_tokens=spec.max_context_tokens,
            workspace=workspace,
            permission_policy=PermissionPolicy(spec.permission_mode),
            approval_callback=approve,
            store=store,
            session_state=session_state,
        )
        if not variant.working_memory_enabled:
            agent.session_state.working_memory.files.clear()
            agent.session_state.working_memory.recent_commands.clear()
            agent.session_state.working_memory.blockers.clear()
            agent.session_state.working_memory.next_steps.clear()
            agent.session_state.working_memory.current_request = ""
            agent.session_state.working_memory.task_summary = ""
            agent.working_memory = _DisabledWorkingMemory()
        if not variant.context_policy_enabled:
            agent.context = _DisabledContextPolicy(spec.max_context_tokens)
        return agent, llm, store

    @staticmethod
    def _tools_for(
        benchmark_id: str,
        workspace: WorkspaceContext,
        probe: _ParallelProbe | None,
    ) -> list[Tool]:
        if benchmark_id == "parallel-read":
            if probe is None:
                raise ValueError("parallel-read requires a probe")
            return [
                _ProbeReadFileTool(probe, workspace),
                _ProbeGrepTool(probe, workspace),
            ]
        if benchmark_id == "large-output":
            return [_DeterministicLargeBashTool(workspace)]
        return [
            tool for tool in create_tools(workspace)
            if tool.name != "agent"
        ]

    def _build_outcome(
        self,
        spec: FixtureSpec,
        variant: AblationConfig,
        repo_root: Path,
        execution: _Execution,
    ) -> BenchmarkOutcome:
        report = self._latest_report(execution.agent, execution.store)
        trace_events = self._latest_trace(execution.agent, execution.store)
        tool_events = [
            event.data for event in trace_events if event.event == "tool_completed"
        ]
        checks = {
            "script_consumed": execution.llm.remaining == 0,
            "tool_pairing": _tool_pairing_valid(execution.agent.messages),
            **execution.extra_checks,
        }
        expected = spec.expected
        if "final_answer" in expected:
            checks["final_answer"] = execution.final_answer == expected["final_answer"]
        if "files_equal" in expected:
            for relative, content in sorted(expected["files_equal"].items()):
                checks[f"file_equal:{relative}"] = (
                    (repo_root / relative).read_text(encoding="utf-8") == content
                )
        if "files_contain" in expected:
            for relative, content in sorted(expected["files_contain"].items()):
                checks[f"file_contains:{relative}"] = content in (
                    repo_root / relative
                ).read_text(encoding="utf-8")
        if "files_absent" in expected:
            for relative in expected["files_absent"]:
                checks[f"file_absent:{relative}"] = not (repo_root / relative).exists()
        if "tool_events" in expected:
            checks["tool_events"] = _tool_events_match(
                tool_events,
                expected["tool_events"],
            )
        if "tool_result_contains" in expected:
            results = {
                message.get("tool_call_id"): str(message.get("content", ""))
                for message in execution.agent.messages
                if message.get("role") == "tool"
            }
            for call_id, content in expected["tool_result_contains"].items():
                checks[f"tool_result:{call_id}"] = content in results.get(call_id, "")
        if "approval_count" in expected:
            checks["approval_count"] = (
                report is not None
                and report.approval_count == int(expected["approval_count"])
            )
        if "compression_min" in expected:
            expected_compressions = int(expected["compression_min"])
            checks["compression_min"] = (
                report is not None
                and report.context_compressions >= expected_compressions
            )
            compression_events = [
                event.data
                for event in trace_events
                if event.event == "context_compressed"
            ]
            checks["compression_trace"] = (
                len(compression_events) >= expected_compressions
                and all(
                    event.get("strategy")
                    and event.get("after_tokens", 0) < event.get("before_tokens", 0)
                    for event in compression_events
                )
            )
        if "recovery_status" in expected:
            checks["recovery_status"] = (
                execution.recovery_status == expected["recovery_status"]
            )
        if "working_memory_file" in expected:
            expected_file = expected["working_memory_file"]
            remembered = next((
                item for item in execution.agent.session_state.working_memory.files
                if item.path == expected_file["path"]
            ), None)
            checks["working_memory_file"] = bool(
                remembered is not None
                and remembered.action == expected_file["action"]
                and remembered.fresh is expected_file["fresh"]
            )

        failure_categories = {
            str(event.get("error_code") or event.get("status"))
            for event in tool_events
            if event.get("status") != "ok"
        }
        if execution.exception_type:
            failure_categories.add(f"exception:{execution.exception_type}")
        failure_categories.update(
            f"check:{name}" for name, passed in checks.items() if not passed
        )
        passed = execution.exception_type is None and all(checks.values())
        return BenchmarkOutcome(
            benchmark_id=spec.benchmark_id,
            variant=variant.name,
            working_memory_enabled=variant.working_memory_enabled,
            context_policy_enabled=variant.context_policy_enabled,
            passed=passed,
            completed=execution.completed,
            model_attempts=report.model_attempts if report is not None else 0,
            tool_steps=report.tool_steps if report is not None else 0,
            prompt_tokens=report.prompt_tokens if report is not None else 0,
            completion_tokens=report.completion_tokens if report is not None else 0,
            approval_count=report.approval_count if report is not None else 0,
            compression_count=(report.context_compressions if report is not None else 0),
            recovery_status=execution.recovery_status,
            stop_reason=execution.stop_reason,
            failure_categories=tuple(sorted(failure_categories)),
            checks=dict(sorted(checks.items())),
        )

    @staticmethod
    def _latest_report(agent: Agent, store: ProjectStore):
        if not agent.session_state.run_ids:
            return None
        return store.load_report(agent.session_state.run_ids[-1])

    @staticmethod
    def _latest_trace(agent: Agent, store: ProjectStore):
        if not agent.session_state.run_ids:
            return []
        return store.read_trace(agent.session_state.run_ids[-1]).events


def _tool_pairing_valid(messages: list[dict]) -> bool:
    expected_ids = []
    expected_indexes = []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        call_ids = [call.get("id") for call in calls]
        replies = messages[index + 1:index + 1 + len(call_ids)]
        if (
            message.get("role") != "assistant"
            or len(replies) != len(call_ids)
            or [reply.get("role") for reply in replies] != ["tool"] * len(call_ids)
            or [reply.get("tool_call_id") for reply in replies] != call_ids
        ):
            return False
        expected_ids.extend(call_ids)
        expected_indexes.extend(range(index + 1, index + 1 + len(call_ids)))
    actual_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    ]
    actual_ids = [messages[index].get("tool_call_id") for index in actual_indexes]
    return (
        actual_indexes == expected_indexes
        and actual_ids == expected_ids
        and len(expected_ids) == len(set(expected_ids))
    )


def _tool_events_match(observed: list[dict], expected: list[dict]) -> bool:
    if len(observed) != len(expected):
        return False
    return all(
        all(actual.get(key) == value for key, value in wanted.items())
        for actual, wanted in zip(observed, expected)
    )
