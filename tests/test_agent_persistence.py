"""Agent durability points exercised entirely with a scripted FakeLLM."""

import copy

import pytest

from pikacore.agent import Agent
from pikacore.checkpoint import INTERRUPTED_TOOL_RESULT
from pikacore.llm import LLMResponse, ToolCall
from pikacore.permissions import PermissionPolicy
from pikacore.state import RunState, SessionState
from pikacore.store import ProjectStore, read_json
from pikacore.tools.agent import AgentTool
from pikacore.tools.base import Tool
from pikacore.tools.write import WriteFileTool
from pikacore.workspace import WorkspaceContext
from tests.protocol_assertions import assert_tool_pairing


class FakeLLM:
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("FakeLLM has no response")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        if on_token and response.content:
            on_token(response.content)
        return response


class EchoTool(Tool):
    name = "echo"
    description = "Echo a value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, value: str) -> str:
        return f"echo:{value}"


class InterruptTool(Tool):
    name = "interrupt"
    description = "Interrupt execution."
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        raise KeyboardInterrupt


class LongOutputTool(Tool):
    name = "long_output"
    description = "Return compressible output."
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        return "\n".join(f"line {index}: " + "x" * 40 for index in range(100))


class RecordingStore(ProjectStore):
    def __init__(self, state_root):
        super().__init__(state_root=state_root)
        self.session_snapshots: list[dict] = []

    def save_session(self, state: SessionState) -> None:
        super().save_session(state)
        self.session_snapshots.append(read_json(self.session_path(state.session_id)))


class FailingOnceStore(ProjectStore):
    def __init__(self, state_root):
        super().__init__(state_root=state_root)
        self.fail_next_session_save = True

    def save_session(self, state: SessionState) -> None:
        if self.fail_next_session_save:
            self.fail_next_session_save = False
            raise OSError("simulated durability failure")
        super().save_session(state)


class FailingFinalSessionStore(ProjectStore):
    def __init__(self, state_root):
        super().__init__(state_root=state_root)
        self.session_save_count = 0

    def save_session(self, state: SessionState) -> None:
        self.session_save_count += 1
        if self.session_save_count == 4:
            raise OSError("simulated final session failure")
        super().save_session(state)


def _tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(tool_calls=list(calls), prompt_tokens=10, completion_tokens=2)


def _final_response(content="done") -> LLMResponse:
    return LLMResponse(content=content, prompt_tokens=7, completion_tokens=3)


def test_agent_persists_every_message_boundary_trace_run_and_report(tmp_path):
    store = RecordingStore(tmp_path / "state")
    llm = FakeLLM([
        _tool_response(
            ToolCall("call-a", "echo", {"value": "a"}),
            ToolCall("call-b", "echo", {"value": "b"}),
        ),
        _final_response(),
    ])
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    assert agent.chat("persist this") == "done"
    assert_tool_pairing(agent.messages)

    role_snapshots = [
        [message["role"] for message in snapshot["messages"]]
        for snapshot in store.session_snapshots
    ]
    assert ["user"] in role_snapshots
    assert ["user", "assistant"] in role_snapshots
    assert ["user", "assistant", "tool"] in role_snapshots
    assert ["user", "assistant", "tool", "tool"] in role_snapshots
    assert role_snapshots[-1] == ["user", "assistant", "tool", "tool", "assistant"]

    run_id = agent.session_state.run_ids[-1]
    persisted_session = store.load_session(agent.session_state.session_id)
    persisted_run = store.load_run(run_id)
    report = store.load_report(run_id)
    trace = store.read_trace(run_id)

    assert persisted_session is not None
    assert persisted_session.messages == agent.messages
    assert persisted_run == RunState.from_dict(read_json(store.task_state_path(run_id)))
    assert persisted_run.status == "completed"
    assert persisted_run.stop_reason == "completed"
    assert persisted_run.model_attempts == 2
    assert persisted_run.tool_steps == 2
    assert report is not None
    assert report.completed is True
    assert report.prompt_tokens == 17
    assert report.completion_tokens == 5
    assert report.tool_calls == {"echo": 2}
    assert report.tool_steps == 2
    assert trace.warnings == []
    assert [event.seq for event in trace.events] == list(range(1, len(trace.events) + 1))
    event_names = [event.event for event in trace.events]
    assert event_names[0] == "run_started"
    assert event_names.count("message_appended") == 5
    assert event_names.count("tool_requested") == 2
    assert event_names.count("tool_completed") == 2
    assert event_names[-1] == "run_finished"


def test_interrupt_backfill_and_run_failure_are_persisted_before_reraise(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    llm = FakeLLM([
        _tool_response(
            ToolCall("interrupt-id", "interrupt", {}),
            ToolCall("pending-id", "echo", {"value": "pending"}),
        )
    ])
    agent = Agent(
        llm=llm,
        tools=[InterruptTool(), EchoTool()],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    with pytest.raises(KeyboardInterrupt):
        agent.chat("interrupt")

    assert_tool_pairing(agent.messages)
    assert [
        (message["tool_call_id"], message["content"])
        for message in agent.messages
        if message.get("role") == "tool"
    ] == [
        ("interrupt-id", INTERRUPTED_TOOL_RESULT),
        ("pending-id", INTERRUPTED_TOOL_RESULT),
    ]
    run_id = agent.session_state.run_ids[-1]
    persisted_session = store.load_session(agent.session_state.session_id)
    persisted_run = store.load_run(run_id)
    report = store.load_report(run_id)
    assert persisted_session is not None
    assert persisted_session.messages == agent.messages
    assert persisted_run is not None
    assert persisted_run.status == "interrupted"
    assert persisted_run.stop_reason == "user_interrupted"
    assert report is not None
    assert report.completed is False
    assert report.stop_reason == "user_interrupted"
    assert store.read_trace(run_id).events[-1].event == "run_failed"


def test_context_compression_rewrite_is_saved_and_traced(tmp_path):
    store = RecordingStore(tmp_path / "state")
    llm = FakeLLM([
        _tool_response(ToolCall("long", "long_output", {})),
        _final_response(),
    ])
    agent = Agent(
        llm=llm,
        tools=[LongOutputTool()],
        max_context_tokens=100,
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    assert agent.chat("compress") == "done"

    tool_contents = [
        message["content"]
        for snapshot in store.session_snapshots
        for message in snapshot["messages"]
        if message.get("role") == "tool"
    ]
    assert any("line 50" in content for content in tool_contents)
    assert any("snipped to save context" in content for content in tool_contents)
    run_id = agent.session_state.run_ids[-1]
    report = store.load_report(run_id)
    assert report is not None
    assert report.context_compressions == 1
    assert "context_compressed" in [
        event.event for event in store.read_trace(run_id).events
    ]


def test_model_failure_persists_failed_run_report_and_user_message(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=FakeLLM([RuntimeError("model unavailable")]),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        agent.chat("keep this request")

    run_id = agent.session_state.run_ids[-1]
    session = store.load_session(agent.session_state.session_id)
    run = store.load_run(run_id)
    report = store.load_report(run_id)
    assert session is not None
    assert session.messages == [{"role": "user", "content": "keep this request"}]
    assert run is not None
    assert run.status == "failed"
    assert run.stop_reason == "model_error"
    assert report is not None
    assert report.stop_reason == "model_error"
    assert "RuntimeError: model unavailable" in report.error
    assert store.read_trace(run_id).events[-1].event == "run_failed"


def test_executor_callback_failure_persists_protocol_backfill(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=FakeLLM([
            _tool_response(ToolCall("pending", "echo", {"value": "x"}))
        ]),
        tools=[EchoTool()],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    def fail_callback(_name, _arguments):
        raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError, match="renderer failed"):
        agent.chat("trigger callback", on_tool=fail_callback)

    assert_tool_pairing(agent.messages)
    assert agent.messages[-1] == {
        "role": "tool",
        "tool_call_id": "pending",
        "content": INTERRUPTED_TOOL_RESULT,
    }
    run_id = agent.session_state.run_ids[-1]
    persisted = store.load_session(agent.session_state.session_id)
    report = store.load_report(run_id)
    assert persisted is not None
    assert persisted.messages == agent.messages
    assert report is not None
    assert report.stop_reason == "internal_error"


def test_persistence_failure_warns_and_is_recorded_in_report(tmp_path):
    store = FailingOnceStore(tmp_path / "state")
    agent = Agent(
        llm=FakeLLM([_final_response()]),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    with pytest.warns(RuntimeWarning, match="save session failed"):
        assert agent.chat("continue with warning") == "done"

    report = store.load_report(agent.session_state.run_ids[-1])
    assert report is not None
    assert len(report.persistence_errors) == 1
    assert "simulated durability failure" in report.persistence_errors[0]


def test_final_session_failure_is_recorded_in_persisted_report(tmp_path):
    store = FailingFinalSessionStore(tmp_path / "state")
    agent = Agent(
        llm=FakeLLM([_final_response()]),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    with pytest.warns(RuntimeWarning, match="simulated final session failure"):
        assert agent.chat("finish despite warning") == "done"

    report = store.load_report(agent.session_state.run_ids[-1])
    assert report is not None
    assert len(report.persistence_errors) == 1
    assert "simulated final session failure" in report.persistence_errors[0]


def test_sub_agent_uses_independent_session_and_parent_linked_trace(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    llm = FakeLLM([
        _tool_response(ToolCall("delegate", "agent", {"task": "inspect"})),
        _final_response("child result"),
        _final_response("parent result"),
    ])
    agent = Agent(
        llm=llm,
        tools=[AgentTool()],
        workspace=WorkspaceContext(tmp_path),
        permission_policy=PermissionPolicy("auto"),
        store=store,
    )

    assert agent.chat("delegate work") == "parent result"

    session_paths = sorted(store.sessions_dir.glob("*.json"))
    assert len(session_paths) == 2
    child_path = next(
        path for path in session_paths if path.stem != agent.session_state.session_id
    )
    child_session = store.load_session(child_path.stem)
    assert child_session is not None
    assert child_session.session_id != agent.session_state.session_id
    assert child_session.messages[0] == {"role": "user", "content": "inspect"}
    parent_run_id = agent.session_state.run_ids[-1]
    child_run_id = child_session.run_ids[-1]
    child_started = store.read_trace(child_run_id).events[0]
    assert child_started.event == "run_started"
    assert child_started.data["parent_run_id"] == parent_run_id
    parent_report = store.load_report(parent_run_id)
    child_report = store.load_report(child_run_id)
    assert parent_report is not None
    assert child_report is not None
    assert (parent_report.prompt_tokens, parent_report.completion_tokens) == (17, 5)
    assert (child_report.prompt_tokens, child_report.completion_tokens) == (7, 3)
    assert (
        parent_report.prompt_tokens + child_report.prompt_tokens,
        parent_report.completion_tokens + child_report.completion_tokens,
    ) == (24, 8)


def test_loaded_session_continues_identity_and_run_history(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    original = SessionState(
        session_id="resume-me",
        repo_root=str(tmp_path),
        model="fake-model",
        messages=[{"role": "user", "content": "earlier"}],
        run_ids=["run-earlier"],
    )
    store.save_session(original)
    loaded = store.load_session("resume-me")
    assert loaded is not None
    agent = Agent(
        llm=FakeLLM([_final_response("continued")]),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        store=store,
        session_state=loaded,
    )

    assert agent.chat("continue") == "continued"

    persisted = store.load_session("resume-me")
    assert persisted is not None
    assert persisted.session_id == "resume-me"
    assert persisted.created_at == original.created_at
    assert persisted.run_ids[0] == "run-earlier"
    assert len(persisted.run_ids) == 2
    assert list(store.sessions_dir.glob("*.json")) == [store.session_path("resume-me")]


@pytest.mark.parametrize(
    ("approved", "expected_status", "expected_errors", "file_exists"),
    [
        (True, "tool_approved", 0, True),
        (False, "tool_rejected", 1, False),
    ],
)
def test_report_and_trace_summarize_tool_approval_outcomes(
    tmp_path,
    approved,
    expected_status,
    expected_errors,
    file_exists,
):
    store = ProjectStore(state_root=tmp_path / "state")
    llm = FakeLLM([
        _tool_response(
            ToolCall(
                "write",
                "write_file",
                {"file_path": "output.txt", "content": "saved\n"},
            )
        ),
        _final_response(),
    ])
    agent = Agent(
        llm=llm,
        tools=[WriteFileTool()],
        workspace=WorkspaceContext(tmp_path),
        permission_policy=PermissionPolicy("ask"),
        approval_callback=lambda _tool, _arguments: approved,
        store=store,
    )

    assert agent.chat("write") == "done"

    run_id = agent.session_state.run_ids[-1]
    report = store.load_report(run_id)
    assert report is not None
    assert report.tool_calls == {"write_file": 1}
    assert report.tool_approvals == {"write_file": 1}
    assert report.approval_count == 1
    assert report.tool_error_count == expected_errors
    assert (tmp_path / "output.txt").exists() is file_exists
    if approved:
        assert report.tool_errors == {}
        assert report.affected_paths == [str((tmp_path / "output.txt").resolve())]
    else:
        assert report.tool_errors == {"write_file": 1}
    assert expected_status in [event.event for event in store.read_trace(run_id).events]
