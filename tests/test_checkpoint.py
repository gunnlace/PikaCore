"""Phase 4 checkpoint creation and deterministic recovery tests."""

import copy
import json

import pytest

from pikacore.agent import Agent, CheckpointPersistenceError
from pikacore.checkpoint import (
    INTERRUPTED_TOOL_RESULT,
    UNVERIFIABLE_FRESHNESS,
    apply_recovery,
    build_runtime_identity,
    evaluate_recovery,
)
from pikacore.llm import LLMResponse, ToolCall
from pikacore.permissions import PermissionPolicy
from pikacore.state import Checkpoint, SessionState
from pikacore.store import ProjectStore
from pikacore.tools.base import Tool
from pikacore.tools.glob_tool import GlobTool
from pikacore.tools.grep import GrepTool
from pikacore.tools.read import ReadFileTool
from pikacore.tools.write import WriteFileTool
from pikacore.workspace import WorkspaceContext
from tests.protocol_assertions import assert_tool_pairing


class FakeLLM:
    model = "fake-model"

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("FakeLLM must not be called during recovery")
        response = self.responses.pop(0)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        return response


class NeverReplayTool(Tool):
    description = "A recovery sentinel that must never execute."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, name: str, *, read_only: bool = False):
        super().__init__()
        self.name = name
        self.read_only = read_only
        self.risk_level = "low" if read_only else "high"
        self.execute_count = 0

    def execute(self) -> str:
        self.execute_count += 1
        raise AssertionError(f"pending tool {self.name} was replayed")


class BarrierReadTool(Tool):
    name = "barrier_read"
    description = "Read before a mutating checkpoint barrier."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self):
        super().__init__()
        self.execute_count = 0

    def execute(self) -> str:
        self.execute_count += 1
        return "read complete"


class CheckpointObservingWriteTool(Tool):
    name = "observing_write"
    description = "Verify the read checkpoint exists before mutation."
    parameters = {"type": "object", "properties": {}, "required": []}
    risk_level = "medium"
    read_only = False

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.saw_read_checkpoint = False
        self.execute_count = 0

    def execute(self) -> str:
        self.execute_count += 1
        for path in self.store.checkpoints_dir.glob("*.json"):
            checkpoint = self.store.load_checkpoint(path.stem)
            if checkpoint and checkpoint.completed_tool_call_ids == ["read"]:
                self.saw_read_checkpoint = True
                break
        if not self.saw_read_checkpoint:
            raise AssertionError("read checkpoint was not durable before write")
        return "write complete"


class AlwaysFailCheckpointStore(ProjectStore):
    def save_checkpoint(self, checkpoint):
        raise OSError("checkpoint storage unavailable")


class FailSecondCheckpointStore(ProjectStore):
    def __init__(self, state_root):
        super().__init__(state_root=state_root)
        self.checkpoint_save_count = 0

    def save_checkpoint(self, checkpoint):
        self.checkpoint_save_count += 1
        if self.checkpoint_save_count == 2:
            raise OSError("read barrier checkpoint failed")
        super().save_checkpoint(checkpoint)


def _assistant_tool_calls(*names: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for index, name in enumerate(names)
        ],
    }


def _runtime(workspace, tools, *, model="fake-model", mode="ask"):
    return build_runtime_identity(
        model=model,
        workspace=workspace,
        tools=tools,
        permission_policy=PermissionPolicy(mode),
    )


def _checkpoint(session, runtime, *, freshness=None, pending=None):
    return Checkpoint(
        checkpoint_id="checkpoint-1",
        session_id=session.session_id,
        run_id="run-1",
        current_user_request="continue",
        pending_tool_calls=pending or [],
        file_freshness=freshness or {},
        runtime_identity=runtime,
    )


def test_runtime_identity_is_stable_complete_and_contains_no_credentials(tmp_path):
    workspace = WorkspaceContext(tmp_path, branch="main")
    tools = [ReadFileTool(workspace)]

    identity = _runtime(workspace, tools, mode="auto")

    assert identity == _runtime(workspace, tools, mode="auto")
    assert set(identity) == {
        "model",
        "repo_root",
        "branch",
        "tool_signature",
        "permission_mode",
        "harness_schema_version",
    }
    assert identity["branch"] == "main"
    assert identity["permission_mode"] == "auto"
    assert "key" not in json.dumps(identity).lower()


def test_recovery_classifies_full_valid_and_files_stale(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("before\n", encoding="utf-8")
    workspace = WorkspaceContext(tmp_path)
    tools = [ReadFileTool(workspace)]
    runtime = _runtime(workspace, tools)
    relative, fingerprint = workspace.fingerprint_path(path)
    session = SessionState(session_id="session-1")
    checkpoint = _checkpoint(
        session,
        runtime,
        freshness={relative: fingerprint},
    )

    valid = evaluate_recovery(
        session,
        checkpoint,
        current_runtime=runtime,
        workspace=workspace,
    )
    assert valid.status == "full-valid"
    assert valid.notice is None

    path.write_text("after\n", encoding="utf-8")
    stale = evaluate_recovery(
        session,
        checkpoint,
        current_runtime=runtime,
        workspace=workspace,
    )
    assert stale.status == "files-stale"
    assert stale.stale_paths == ["source.txt"]
    assert "Re-read stale paths" in stale.notice


@pytest.mark.parametrize("change", ["modify", "delete"])
def test_file_freshness_detects_modified_and_missing_files(tmp_path, change):
    path = tmp_path / "tracked.txt"
    path.write_text("known\n", encoding="utf-8")
    workspace = WorkspaceContext(tmp_path)
    relative, fingerprint = workspace.fingerprint_path(path)
    session = SessionState(session_id="session-1")
    runtime = _runtime(workspace, [])
    checkpoint = _checkpoint(session, runtime, freshness={relative: fingerprint})

    if change == "modify":
        path.write_text("changed\n", encoding="utf-8")
    else:
        path.unlink()

    result = evaluate_recovery(
        session,
        checkpoint,
        current_runtime=runtime,
        workspace=workspace,
    )
    assert result.status == "files-stale"
    assert result.stale_paths == ["tracked.txt"]


def test_runtime_mismatch_reports_model_root_branch_tool_and_permission_changes(tmp_path):
    workspace = WorkspaceContext(tmp_path, branch="main")
    session = SessionState(session_id="session-1")
    saved_runtime = _runtime(workspace, [], model="old-model", mode="auto")
    current_runtime = dict(saved_runtime)
    current_runtime.update({
        "model": "new-model",
        "repo_root": "/different/repo",
        "branch": "feature",
        "tool_signature": "different-tools",
        "permission_mode": "read-only",
    })

    result = evaluate_recovery(
        session,
        _checkpoint(session, saved_runtime),
        current_runtime=current_runtime,
        workspace=workspace,
    )

    assert result.status == "runtime-mismatch"
    assert set(result.runtime_differences) == {
        "model",
        "repo_root",
        "branch",
        "tool_signature",
        "permission_mode",
    }
    assert "Runtime differences require review" in result.notice


def test_missing_checkpoint_is_runtime_mismatch_not_blind_resume(tmp_path):
    workspace = WorkspaceContext(tmp_path)
    session = SessionState(session_id="session-1")

    result = evaluate_recovery(
        session,
        None,
        current_runtime=_runtime(workspace, []),
        workspace=workspace,
    )

    assert result.status == "runtime-mismatch"
    assert result.runtime_differences["checkpoint"]["actual"] == "missing"
    assert apply_recovery(session, result) is True
    assert session.messages[-1]["role"] == "user"
    assert "checkpoint" in session.messages[-1]["content"]


def test_pending_tools_are_backfilled_in_order_without_any_replay(tmp_path):
    workspace = WorkspaceContext(tmp_path)
    side_effect = tmp_path / "already-created.txt"
    side_effect.write_text("partial side effect\n", encoding="utf-8")
    tools = [
        NeverReplayTool("read_file", read_only=True),
        NeverReplayTool("write_file"),
        NeverReplayTool("edit_file"),
        NeverReplayTool("bash"),
    ]
    session = SessionState(
        session_id="resume-pending",
        repo_root=str(tmp_path),
        model="fake-model",
        messages=[
            {"role": "user", "content": "continue"},
            _assistant_tool_calls(
                "read_file", "read_file", "write_file", "edit_file", "bash"
            ),
            {
                "role": "tool",
                "tool_call_id": "call-0",
                "content": "previous read completed",
            },
        ],
    )
    store = ProjectStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=FakeLLM(),
        tools=tools,
        workspace=workspace,
        permission_policy=PermissionPolicy("ask"),
        store=store,
        session_state=session,
    )
    checkpoint = _checkpoint(
        session,
        agent._runtime_identity(),
        pending=[
            {"id": f"call-{index}", "name": name, "arguments": {}}
            for index, name in enumerate(
                ["read_file", "read_file", "write_file", "edit_file", "bash"]
            )
        ],
    )
    store.save_checkpoint(checkpoint)
    session.last_checkpoint_id = checkpoint.checkpoint_id
    store.save_session(session)

    result = agent.recover_session()

    assert result.status == "incomplete-tool-call"
    assert result.pending_tool_names == [
        "read_file",
        "write_file",
        "edit_file",
        "bash",
    ]
    assert result.requires_workspace_inspection is True
    assert [tool.execute_count for tool in tools] == [0, 0, 0, 0]
    assert side_effect.read_text(encoding="utf-8") == "partial side effect\n"
    tool_messages = [
        message for message in agent.messages if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-0",
        "call-1",
        "call-2",
        "call-3",
        "call-4",
    ]
    assert tool_messages[0]["content"] == "previous read completed"
    assert [message["content"] for message in tool_messages[1:]] == [
        INTERRUPTED_TOOL_RESULT,
        INTERRUPTED_TOOL_RESULT,
        INTERRUPTED_TOOL_RESULT,
        INTERRUPTED_TOOL_RESULT,
    ]
    assert "Inspect workspace changes" in agent.messages[-1]["content"]
    assert_tool_pairing(agent.messages)
    persisted = store.load_session(session.session_id)
    assert persisted is not None
    assert persisted.messages == agent.messages
    repaired_messages = copy.deepcopy(agent.messages)

    repeated = agent.recover_session()

    assert repeated.status == "full-valid"
    assert agent.messages == repaired_messages


def test_unknown_checkpoint_schema_refuses_recovery_without_mutation(tmp_path):
    workspace = WorkspaceContext(tmp_path)
    tools = [NeverReplayTool("bash")]
    messages = [_assistant_tool_calls("bash")]
    session = SessionState(
        session_id="future-session",
        messages=copy.deepcopy(messages),
        last_checkpoint_id="future-checkpoint",
    )
    store = ProjectStore(state_root=tmp_path / "state")
    path = store.checkpoint_path("future-checkpoint")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    agent = Agent(
        llm=FakeLLM(),
        tools=tools,
        workspace=workspace,
        store=store,
        session_state=session,
    )

    result = agent.recover_session()

    assert result.status == "schema-mismatch"
    assert result.can_resume is False
    assert session.messages == messages
    assert tools[0].execute_count == 0
    assert json.loads(path.read_text(encoding="utf-8")) == {"schema_version": 2}


def test_agent_checkpoint_chain_covers_pending_read_batch_freshness_and_finish(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original\n", encoding="utf-8")
    workspace = WorkspaceContext(tmp_path)
    store = ProjectStore(state_root=tmp_path / "state")
    llm = FakeLLM([
        LLMResponse(
            tool_calls=[ToolCall("read-1", "read_file", {"file_path": "source.txt"})]
        ),
        LLMResponse(content="done", prompt_tokens=3, completion_tokens=1),
    ])
    agent = Agent(
        llm=llm,
        tools=[ReadFileTool()],
        workspace=workspace,
        store=store,
    )

    assert agent.chat("read it") == "done"

    chain = []
    checkpoint_id = agent.session_state.last_checkpoint_id
    while checkpoint_id is not None:
        checkpoint = store.load_checkpoint(checkpoint_id)
        assert checkpoint is not None
        chain.append(checkpoint)
        checkpoint_id = checkpoint.parent_checkpoint_id
    assert len(chain) == 3
    final, read_batch, pending = chain
    assert final.pending_tool_calls == []
    assert final.last_successful_action == "run_completed"
    assert final.file_freshness.keys() == {"source.txt"}
    assert read_batch.completed_tool_call_ids == ["read-1"]
    assert read_batch.pending_tool_calls == []
    assert [call["name"] for call in pending.pending_tool_calls] == ["read_file"]
    trace_names = [
        event.event
        for event in store.read_trace(agent.session_state.run_ids[-1]).events
    ]
    assert trace_names.count("checkpoint_created") == 3
    assert trace_names[-1] == "run_finished"
    report = store.load_report(agent.session_state.run_ids[-1])
    assert report is not None
    assert report.checkpoint_status == "created"

    agent.llm.responses.append(LLMResponse(content="ordinary follow-up"))
    assert agent.chat("no file tools this time") == "ordinary follow-up"
    inherited = store.load_checkpoint(agent.session_state.last_checkpoint_id)
    assert inherited is not None
    assert inherited.file_freshness.keys() == {"source.txt"}

    source.write_text("externally changed\n", encoding="utf-8")
    resumed_session = store.load_session(agent.session_state.session_id)
    assert resumed_session is not None
    resumed = Agent(
        llm=FakeLLM([LLMResponse(content="continued")]),
        tools=[ReadFileTool()],
        workspace=workspace,
        store=store,
        session_state=resumed_session,
    )
    recovery = resumed.recover_session()
    assert recovery.status == "files-stale"
    assert recovery.stale_paths == ["source.txt"]
    assert resumed.llm.calls == []

    assert resumed.chat("continue after checking") == "continued"
    resumed_report = store.load_report(resumed.session_state.run_ids[-1])
    assert resumed_report is not None
    assert resumed_report.recovery_status == "files-stale"


def test_read_batch_checkpoint_is_durable_before_next_write_barrier(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    write = CheckpointObservingWriteTool(store)
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[
                ToolCall("read", "barrier_read", {}),
                ToolCall("write", "observing_write", {}),
            ]),
            LLMResponse(content="done"),
        ]),
        tools=[BarrierReadTool(), write],
        workspace=WorkspaceContext(tmp_path),
        permission_policy=PermissionPolicy("auto"),
        store=store,
    )

    assert agent.chat("read then write") == "done"
    assert write.saw_read_checkpoint is True


def test_checkpoint_failure_prevents_write_and_fails_run(tmp_path):
    store = AlwaysFailCheckpointStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[
                ToolCall(
                    "write",
                    "write_file",
                    {"file_path": "blocked.txt", "content": "must not exist\n"},
                )
            ])
        ]),
        tools=[WriteFileTool()],
        workspace=WorkspaceContext(tmp_path),
        permission_policy=PermissionPolicy("auto"),
        store=store,
    )

    with pytest.warns(RuntimeWarning, match="save checkpoint failed"):
        with pytest.raises(CheckpointPersistenceError):
            agent.chat("write only with recovery evidence")

    assert not (tmp_path / "blocked.txt").exists()
    assert_tool_pairing(agent.messages)
    assert agent.messages[-1]["content"].startswith("Error: tool was not executed")
    run_id = agent.session_state.run_ids[-1]
    report = store.load_report(run_id)
    assert report is not None
    assert report.completed is False
    assert report.stop_reason == "internal_error"
    assert report.tool_errors == {"write_file": 1}
    completed = [
        event
        for event in store.read_trace(run_id).events
        if event.event == "tool_completed"
    ]
    assert completed[0].data["error_code"] == "checkpoint-failed"


def test_read_barrier_checkpoint_failure_stops_following_write(tmp_path):
    store = FailSecondCheckpointStore(tmp_path / "state")
    read = BarrierReadTool()
    write = CheckpointObservingWriteTool(store)
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[
                ToolCall("read", "barrier_read", {}),
                ToolCall("write", "observing_write", {}),
            ])
        ]),
        tools=[read, write],
        workspace=WorkspaceContext(tmp_path),
        permission_policy=PermissionPolicy("auto"),
        store=store,
    )

    with pytest.warns(RuntimeWarning, match="read barrier checkpoint failed"):
        with pytest.raises(CheckpointPersistenceError):
            agent.chat("stop at the barrier")

    assert read.execute_count == 1
    assert write.execute_count == 0
    assert_tool_pairing(agent.messages)
    report = store.load_report(agent.session_state.run_ids[-1])
    assert report is not None
    assert report.completed is False
    assert report.checkpoint_status == "error"


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (GrepTool(), {"pattern": "needle", "path": "."}),
        (GlobTool(), {"pattern": "*.txt", "path": "."}),
    ],
)
def test_directory_search_tools_report_actual_file_freshness_paths(
    tmp_path, tool, arguments
):
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[ToolCall("search", tool.name, arguments)]),
            LLMResponse(content="done"),
        ]),
        tools=[tool],
        workspace=WorkspaceContext(tmp_path),
        store=ProjectStore(repo_root=tmp_path),
    )

    assert agent.chat("search") == "done"

    checkpoint = agent.store.load_checkpoint(agent.session_state.last_checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.file_freshness.keys() == {".", "source.txt"}
    assert checkpoint.file_freshness["."] == UNVERIFIABLE_FRESHNESS

    (tmp_path / "new-match.txt").write_text("needle\n", encoding="utf-8")
    session = agent.store.load_session(agent.session_state.session_id)
    assert session is not None
    resumed = Agent(
        llm=FakeLLM(),
        tools=[type(tool)()],
        workspace=WorkspaceContext(tmp_path),
        store=agent.store,
        session_state=session,
    )

    recovery = resumed.recover_session()

    assert recovery.status == "files-stale"
    assert "." in recovery.stale_paths
    assert resumed.llm.calls == []


def test_grep_directory_freshness_detects_later_matching_file_change(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("needle = 1\n", encoding="utf-8")
    store = ProjectStore(repo_root=tmp_path)
    workspace = WorkspaceContext(tmp_path)
    agent = Agent(
        llm=FakeLLM([
            LLMResponse(tool_calls=[
                ToolCall("grep", "grep", {"pattern": "needle", "path": "."})
            ]),
            LLMResponse(content="done"),
        ]),
        tools=[GrepTool()],
        workspace=workspace,
        store=store,
    )
    assert agent.chat("find it") == "done"

    source.write_text("needle = 2\n", encoding="utf-8")
    session = store.load_session(agent.session_state.session_id)
    assert session is not None
    resumed = Agent(
        llm=FakeLLM(),
        tools=[GrepTool()],
        workspace=workspace,
        store=store,
        session_state=session,
    )

    recovery = resumed.recover_session()

    assert recovery.status == "files-stale"
    assert recovery.stale_paths == [".", "source.py"]
