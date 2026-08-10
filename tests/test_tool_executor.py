"""Structured execution, approval threading, and read/write barrier tests."""

import subprocess
import threading

import pytest

from pikacore.llm import ToolCall
from pikacore.permissions import PermissionPolicy
from pikacore.tool_executor import ToolExecutor
from pikacore.tools.base import Tool
from pikacore.tools.bash import BashTool
from pikacore.tools.write import WriteFileTool
from pikacore.workspace import WorkspaceContext


class MutatingTool(Tool):
    name = "mutate"
    description = "Record one mutation."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    risk_level = "medium"
    read_only = False

    def __init__(self):
        super().__init__()
        self.values: list[str] = []
        self.thread_ids: list[int] = []

    def execute(self, value: str) -> str:
        self.values.append(value)
        self.thread_ids.append(threading.get_ident())
        return f"mutated:{value}"


class VariadicTool(Tool):
    name = "variadic"
    description = "Record arbitrary keyword arguments."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }

    def __init__(self):
        super().__init__()
        self.received = None

    def execute(self, **kwargs) -> str:
        self.received = kwargs
        return "received"


class AgentMutationTool(Tool):
    name = "agent"
    description = "Simulate a sub-agent mutation with no declared path argument."
    parameters = {
        "type": "object",
        "properties": {"filename": {"type": "string"}},
        "required": ["filename"],
    }
    risk_level = "high"
    read_only = False

    def execute(self, filename: str) -> str:
        (self.workspace_context.repo_root / filename).write_text("from agent\n", encoding="utf-8")
        return "agent complete"


class RaisingMutationTool(Tool):
    name = "raising_mutation"
    description = "Write a file before raising an exception."
    parameters = {
        "type": "object",
        "properties": {"filename": {"type": "string"}},
        "required": ["filename"],
    }
    risk_level = "medium"
    read_only = False

    def execute(self, filename: str) -> str:
        (self.workspace_context.repo_root / filename).write_text("partial\n", encoding="utf-8")
        raise RuntimeError("failed after write")


@pytest.mark.parametrize(
    ("mode", "approval", "expected_status", "expected_approval", "executed"),
    [
        ("read-only", True, "rejected", "rejected", False),
        ("ask", False, "rejected", "rejected", False),
        ("ask", True, "ok", "approved", True),
        ("auto", None, "ok", "not_required", True),
    ],
)
def test_permission_modes_control_mutating_execution(
    mode, approval, expected_status, expected_approval, executed
):
    tool = MutatingTool()
    callback_calls = []

    def approve(callback_tool, arguments):
        callback_calls.append((callback_tool.name, arguments))
        return approval

    executor = ToolExecutor(
        [tool],
        permission_policy=PermissionPolicy(mode),
        approval_callback=approve,
    )

    result = executor.execute_one(ToolCall("call", "mutate", {"value": "x"}))

    assert result.status == expected_status
    assert result.approval == expected_approval
    assert bool(tool.values) is executed
    assert bool(callback_calls) is (mode == "ask")


def test_approval_callback_and_mutation_run_on_main_thread():
    main_thread = threading.get_ident()
    approval_threads = []
    tool = MutatingTool()

    def approve(_tool, _arguments):
        approval_threads.append(threading.get_ident())
        return True

    executor = ToolExecutor(
        [tool],
        permission_policy=PermissionPolicy("ask"),
        approval_callback=approve,
    )

    result = executor.execute_many([ToolCall("call", "mutate", {"value": "x"})])[0]

    assert result.status == "ok"
    assert approval_threads == [main_thread]
    assert tool.thread_ids == [main_thread]


def test_variadic_keyword_arguments_are_not_nested():
    tool = VariadicTool()
    executor = ToolExecutor([tool])

    result = executor.execute_one(ToolCall("call", "variadic", {"value": "x"}))

    assert result.status == "ok"
    assert result.workspace_changed is False
    assert tool.received == {"value": "x"}


class SchedulingRecorder:
    def __init__(self):
        self.events: list[tuple[str, int]] = []
        self.lock = threading.Lock()
        self.read_barrier = threading.Barrier(2)
        self.second_read_finished = threading.Event()

    def record(self, event: str):
        with self.lock:
            self.events.append((event, threading.get_ident()))


class ScheduledReadTool(Tool):
    name = "scheduled_read"
    description = "A read used to observe scheduling."
    parameters = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }

    def __init__(self, recorder: SchedulingRecorder):
        super().__init__()
        self.recorder = recorder

    def execute(self, label: str) -> str:
        self.recorder.record(f"read-{label}-start")
        if label in {"a", "b"}:
            self.recorder.read_barrier.wait(timeout=2)
            if label == "a":
                if not self.recorder.second_read_finished.wait(timeout=2):
                    raise AssertionError("second read did not finish")
            else:
                self.recorder.record("read-b-end")
                self.recorder.second_read_finished.set()
                return "read:b"
        self.recorder.record(f"read-{label}-end")
        return f"read:{label}"


class ScheduledBarrierTool(Tool):
    description = "A mutating barrier used to observe scheduling."
    parameters = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }
    risk_level = "high"
    read_only = False

    def __init__(self, name: str, recorder: SchedulingRecorder):
        super().__init__()
        self.name = name
        self.recorder = recorder

    def execute(self, label: str) -> str:
        self.recorder.record(f"{self.name}-{label}-start")
        self.recorder.record(f"{self.name}-{label}-end")
        return f"{self.name}:{label}"


def test_read_batches_run_in_parallel_and_mutations_form_serial_barriers():
    main_thread = threading.get_ident()
    recorder = SchedulingRecorder()
    read = ScheduledReadTool(recorder)
    write = ScheduledBarrierTool("write", recorder)
    bash = ScheduledBarrierTool("bash", recorder)
    executor = ToolExecutor(
        [read, write, bash],
        permission_policy=PermissionPolicy("auto"),
    )
    calls = [
        ToolCall("read-a", "scheduled_read", {"label": "a"}),
        ToolCall("read-b", "scheduled_read", {"label": "b"}),
        ToolCall("write", "write", {"label": "w"}),
        ToolCall("read-c", "scheduled_read", {"label": "c"}),
        ToolCall("bash", "bash", {"label": "x"}),
    ]

    results = executor.execute_many(calls)

    assert [result.tool_call_id for result in results] == [call.id for call in calls]
    assert [result.content for result in results] == [
        "read:a",
        "read:b",
        "write:w",
        "read:c",
        "bash:x",
    ]

    event_names = [event for event, _thread in recorder.events]
    positions = {event: event_names.index(event) for event in event_names}
    assert positions["read-a-start"] < positions["read-b-end"]
    assert positions["read-b-start"] < positions["read-a-end"]
    assert max(positions["read-a-end"], positions["read-b-end"]) < positions["write-w-start"]
    assert positions["write-w-end"] < positions["read-c-start"]
    assert positions["read-c-end"] < positions["bash-x-start"]

    event_threads = dict(recorder.events)
    assert event_threads["write-w-start"] == main_thread
    assert event_threads["bash-x-start"] == main_thread
    assert event_threads["read-a-start"] != main_thread
    assert event_threads["read-b-start"] != main_thread


def test_rejected_barrier_stops_consecutive_risky_calls_but_not_later_reads():
    recorder = SchedulingRecorder()
    read = ScheduledReadTool(recorder)
    first_write = ScheduledBarrierTool("write", recorder)
    second_write = ScheduledBarrierTool("bash", recorder)
    approvals = iter([False, True])
    executor = ToolExecutor(
        [read, first_write, second_write],
        permission_policy=PermissionPolicy("ask"),
        approval_callback=lambda _tool, _arguments: next(approvals),
    )
    calls = [
        ToolCall("denied", "write", {"label": "first"}),
        ToolCall("barrier", "bash", {"label": "second"}),
        ToolCall("read", "scheduled_read", {"label": "c"}),
        ToolCall("approved", "write", {"label": "third"}),
    ]

    results = executor.execute_many(calls)

    assert [result.status for result in results] == ["rejected", "rejected", "ok", "ok"]
    assert [result.error_code for result in results[:2]] == [
        "approval-rejected",
        "rejected-by-barrier",
    ]
    assert "read-c-start" in [event for event, _thread in recorder.events]
    assert "write-third-start" in [event for event, _thread in recorder.events]
    assert "bash-second-start" not in [event for event, _thread in recorder.events]


def test_structured_result_reports_canonical_affected_path(tmp_path):
    workspace = WorkspaceContext(tmp_path)
    tool = WriteFileTool(workspace=workspace)
    executor = ToolExecutor(
        [tool],
        permission_policy=PermissionPolicy("auto"),
    )

    result = executor.execute_one(
        ToolCall("write", "write_file", {"file_path": "new/file.txt", "content": "ok\n"})
    )

    assert result.status == "ok"
    assert result.affected_paths == [str((tmp_path / "new" / "file.txt").resolve())]
    assert result.workspace_changed is True
    assert result.read_paths == []


def test_structured_result_rejects_path_before_tool_execution(tmp_path):
    workspace = WorkspaceContext(tmp_path / "repo")
    workspace.repo_root.mkdir()
    tool = WriteFileTool(workspace=workspace)
    executor = ToolExecutor(
        [tool],
        permission_policy=PermissionPolicy("auto"),
    )

    result = executor.execute_one(
        ToolCall("write", "write_file", {"file_path": "../outside.txt", "content": "blocked\n"})
    )

    assert result.status == "error"
    assert result.error_code == "path-rejected"
    assert not (tmp_path / "outside.txt").exists()


def test_auto_mode_does_not_bypass_bash_hard_deny(tmp_path):
    bash = BashTool(workspace=WorkspaceContext(tmp_path))
    executor = ToolExecutor(
        [bash],
        permission_policy=PermissionPolicy("auto"),
    )

    result = executor.execute_one(ToolCall("bash", "bash", {"command": "rm -rf /"}))

    assert result.status == "error"
    assert "Blocked" in result.content


def test_bash_nonzero_exit_is_structured_failure(tmp_path):
    bash = BashTool(workspace=WorkspaceContext(tmp_path))
    executor = ToolExecutor(
        [bash],
        permission_policy=PermissionPolicy("auto"),
    )

    result = executor.execute_one(ToolCall("bash", "bash", {"command": "exit 7"}))

    assert result.status == "error"
    assert result.error_code == "nonzero-exit"
    assert result.exit_code == 7


def test_bash_and_agent_mutations_are_detected_without_path_arguments(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    workspace = WorkspaceContext(tmp_path)
    bash = BashTool(workspace=workspace)
    agent = AgentMutationTool(workspace=workspace)
    executor = ToolExecutor(
        [bash, agent],
        permission_policy=PermissionPolicy("auto"),
    )

    bash_result = executor.execute_one(
        ToolCall("bash", "bash", {"command": "printf bash > changed-by-bash.txt"})
    )
    agent_result = executor.execute_one(
        ToolCall("agent", "agent", {"filename": "changed-by-agent.txt"})
    )

    assert bash_result.workspace_changed is True
    assert str((tmp_path / "changed-by-bash.txt").resolve()) in bash_result.affected_paths
    assert agent_result.workspace_changed is True
    assert str((tmp_path / "changed-by-agent.txt").resolve()) in agent_result.affected_paths


def test_workspace_delta_does_not_attribute_preexisting_dirty_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "unrelated.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=PikaCore Tests",
            "-c",
            "user.email=tests@pikacore.local",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    unrelated.write_text("user change\n", encoding="utf-8")
    workspace = WorkspaceContext(tmp_path)
    executor = ToolExecutor(
        [BashTool(workspace=workspace)],
        permission_policy=PermissionPolicy("auto"),
    )

    result = executor.execute_one(
        ToolCall("bash", "bash", {"command": "printf tool > created.txt"})
    )
    dirty_result = executor.execute_one(
        ToolCall("bash-2", "bash", {"command": "printf newer > unrelated.txt"})
    )

    assert result.workspace_changed is True
    assert result.affected_paths == [str((tmp_path / "created.txt").resolve())]
    assert dirty_result.workspace_changed is True
    assert dirty_result.affected_paths == [str(unrelated.resolve())]


def test_non_git_bash_and_agent_mutations_are_conservatively_visible(tmp_path):
    workspace = WorkspaceContext(tmp_path)
    executor = ToolExecutor(
        [BashTool(workspace=workspace), AgentMutationTool(workspace=workspace)],
        permission_policy=PermissionPolicy("auto"),
    )

    bash_result = executor.execute_one(
        ToolCall("bash", "bash", {"command": "printf bash > changed-by-bash.txt"})
    )
    agent_result = executor.execute_one(
        ToolCall("agent", "agent", {"filename": "changed-by-agent.txt"})
    )

    assert bash_result.workspace_changed is True
    assert agent_result.workspace_changed is True


def test_non_git_nonzero_bash_reports_possible_partial_side_effects(tmp_path):
    workspace = WorkspaceContext(tmp_path)
    executor = ToolExecutor(
        [BashTool(workspace=workspace)],
        permission_policy=PermissionPolicy("auto"),
    )

    result = executor.execute_one(
        ToolCall(
            "bash",
            "bash",
            {"command": "printf partial > touched.txt; exit 7"},
        )
    )

    assert result.status == "error"
    assert result.exit_code == 7
    assert result.workspace_changed is True
    assert (tmp_path / "touched.txt").read_text(encoding="utf-8") == "partial"


def test_non_git_raising_mutation_reports_possible_partial_side_effects(tmp_path):
    workspace = WorkspaceContext(tmp_path)
    tool = RaisingMutationTool(workspace=workspace)
    executor = ToolExecutor(
        [tool],
        permission_policy=PermissionPolicy("auto"),
    )

    result = executor.execute_one(
        ToolCall("raising", "raising_mutation", {"filename": "touched.txt"})
    )

    assert result.status == "error"
    assert result.error_code == "tool-error"
    assert result.workspace_changed is True
    assert (tmp_path / "touched.txt").read_text(encoding="utf-8") == "partial\n"
