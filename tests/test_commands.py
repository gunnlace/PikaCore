"""CLI command routing over public state APIs; no terminal or provider calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pikacore import commands
from pikacore.agent import Agent
from pikacore.checkpoint import RecoveryResult
from pikacore.commands import execute_command
from pikacore.config import Config
from pikacore.context import CompressionResult
from pikacore.permissions import PermissionPolicy
from pikacore.state import (
    FileMemory,
    Report,
    RunState,
    SessionState,
    TraceEvent,
    WorkingMemory,
)
from pikacore.store import ProjectStore
from pikacore.workspace import WorkspaceContext


class FakeAgent:
    def __init__(self):
        self.llm = SimpleNamespace(
            model="fake-model",
            total_prompt_tokens=12,
            total_completion_tokens=5,
            estimated_cost=0.25,
        )
        self.session_state = SessionState(
            session_id="session-active",
            repo_root="/repo",
            model="fake-model",
            messages=[{"role": "user", "content": "keep me"}],
            working_memory=WorkingMemory(
                current_request="fix it",
                files=[
                    FileMemory(
                        path="src/app.py",
                        action="read",
                        summary="entry point",
                        fresh=False,
                        updated_at="2026-01-01T00:00:00+00:00",
                    )
                ],
            ),
            run_ids=["run-1"],
        )
        self.permission_policy = PermissionPolicy("ask")
        self.tools = [
            SimpleNamespace(
                name="read_file",
                risk_level="low",
                read_only=True,
            ),
            SimpleNamespace(
                name="write_file",
                risk_level="medium",
                read_only=False,
            ),
        ]
        self.calls: list[tuple] = []

    @property
    def messages(self):
        return self.session_state.messages

    def reset(self):
        self.calls.append(("reset",))
        self.messages.clear()
        self.session_state.working_memory = WorkingMemory()

    def clear_working_memory(self):
        self.calls.append(("clear_working_memory",))
        self.session_state.working_memory = WorkingMemory()

    def new_session(self):
        self.calls.append(("new_session",))
        self.session_state = SessionState(
            session_id="session-new",
            repo_root="/repo",
            model=self.llm.model,
        )
        return self.session_state

    def resume_session(self, session_id):
        self.calls.append(("resume_session", session_id))
        if session_id == "missing":
            return None
        self.session_state.session_id = session_id
        return RecoveryResult(status="full-valid", notice="Recovery checked.")

    def recent_runs(self, limit):
        self.calls.append(("recent_runs", limit))
        return [
            RunState(
                run_id="run-1",
                session_id=self.session_state.session_id,
                status="completed",
                stop_reason="completed",
                model_attempts=2,
                tool_steps=1,
            )
        ]

    def recent_trace(self, run_id, limit):
        self.calls.append(("recent_trace", run_id, limit))
        return [
            TraceEvent(
                seq=3,
                event="run_finished",
                session_id=self.session_state.session_id,
                run_id=run_id or "run-1",
                data={"status": "completed"},
            )
        ], ["ignored final line"]

    def set_model(self, model):
        self.calls.append(("set_model", model))
        self.llm.model = model
        self.session_state.model = model
        return model

    def set_permission_mode(self, mode):
        self.calls.append(("set_permission_mode", mode))
        self.permission_policy = PermissionPolicy(mode)
        return self.permission_policy

    def compact_context(self):
        self.calls.append(("compact_context",))
        return CompressionResult(
            changed=True,
            strategy="summary",
            before_tokens=100,
            after_tokens=40,
            removed_messages=2,
            summarized_messages=2,
        )

    def session_token_usage(self):
        self.calls.append(("session_token_usage",))
        return 12, 5, 0.25

    def modified_paths(self):
        self.calls.append(("modified_paths",))
        return ["a.py", "b.py"]

    def save_session_snapshot(self, name=None):
        self.calls.append(("save_session_snapshot", name))
        return name or "snapshot"


@pytest.fixture
def fake_agent():
    return FakeAgent()


@pytest.fixture(autouse=True)
def forbid_terminal_input(monkeypatch):
    def fail_input(*_args, **_kwargs):
        raise AssertionError("command parsing must not call input()")

    monkeypatch.setattr("builtins.input", fail_input)


def run(command, fake_agent, *, confirm=None):
    return execute_command(
        command,
        agent=fake_agent,
        config=Config(model="fake-model"),
        confirm=confirm,
    )


def test_non_command_is_not_handled_and_local_exit_is_handled(fake_agent):
    assert run("please fix this", fake_agent).handled is False
    assert run("quit", fake_agent).exit_requested is True
    assert run("/exit", fake_agent).exit_requested is True


def test_memory_summary_files_and_confirmed_clear(fake_agent):
    summary = run("/memory", fake_agent)
    assert "fix it" in summary.lines[0]

    files = run("/memory files", fake_agent)
    assert files.lines == (
        "src/app.py: action=read, fresh=no, updated=2026-01-01T00:00:00+00:00",
    )

    cancelled = run("/memory clear", fake_agent, confirm=lambda _message: False)
    assert "cancelled" in cancelled.lines[0]
    assert fake_agent.messages == [{"role": "user", "content": "keep me"}]
    assert fake_agent.session_state.working_memory.files

    prompts = []
    cleared = run(
        "/memory clear",
        fake_agent,
        confirm=lambda message: prompts.append(message) or True,
    )
    assert "messages were kept" in cleared.lines[0]
    assert prompts == ["Clear Working Memory? Conversation messages will be kept."]
    assert fake_agent.messages == [{"role": "user", "content": "keep me"}]
    assert fake_agent.session_state.working_memory.files == []


def test_session_commands_and_sessions_alias(fake_agent, monkeypatch):
    listed = [{
        "id": "saved-one",
        "model": "m",
        "saved_at": "now",
        "preview": "hello",
    }]
    monkeypatch.setattr(commands, "list_sessions", lambda: listed)

    current = run("/session", fake_agent)
    assert current.lines[0] == "Session: session-active"
    assert run("/session list", fake_agent).lines == (
        "saved-one (m, now) hello",
    )
    assert run("/sessions", fake_agent).lines == (
        "saved-one (m, now) hello",
    )
    assert run("/session new", fake_agent).lines == ("New session: session-new",)

    resumed = run("/session resume saved-one", fake_agent)
    assert resumed.lines == (
        "Resumed saved-one: recovery=full-valid.",
        "Recovery checked.",
    )
    assert ("resume_session", "saved-one") in fake_agent.calls
    assert run("/session resume missing", fake_agent).lines == (
        "Session not found: missing",
    )


def test_runs_and_trace_parse_defaults_explicit_values_and_errors(fake_agent):
    runs = run("/runs", fake_agent)
    assert runs.lines[0].startswith("run-1: completed")
    assert ("recent_runs", 10) in fake_agent.calls

    run("/runs 3", fake_agent)
    assert ("recent_runs", 3) in fake_agent.calls
    assert run("/runs 0", fake_agent).lines == ("Usage: /runs [n]",)

    trace = run("/trace", fake_agent)
    assert trace.lines[0].startswith("#3 ")
    assert trace.lines[-1] == "Warning: ignored final line"
    assert ("recent_trace", None, 20) in fake_agent.calls

    run("/trace run-special 7", fake_agent)
    assert ("recent_trace", "run-special", 7) in fake_agent.calls
    run("/trace 5", fake_agent)
    assert ("recent_trace", None, 5) in fake_agent.calls
    assert run("/trace run-special nope", fake_agent).lines == (
        "Usage: /trace [run_id] [n]",
    )


def test_permissions_display_and_runtime_change(fake_agent):
    shown = run("/permissions", fake_agent)
    assert shown.lines == (
        "Permission mode: ask",
        "read_file: risk=low, scope=read-only",
        "write_file: risk=medium, scope=mutating",
    )

    changed = run("/permissions auto", fake_agent)
    assert changed.lines == ("Permission mode changed to auto.",)
    assert fake_agent.calls[-1] == ("set_permission_mode", "auto")
    assert run("/permissions unsafe", fake_agent).lines == (
        "Usage: /permissions [read-only|ask|auto]",
    )


def test_retained_commands_route_through_agent_and_session_apis(
    fake_agent,
):
    assert "17 total" in run("/tokens", fake_agent).lines[0]
    assert run("/model", fake_agent).lines == ("Current model: fake-model",)
    assert run("/model next-model", fake_agent).lines == (
        "Switched to next-model.",
    )
    assert fake_agent.calls[-1] == ("set_model", "next-model")

    compact = run("/compact", fake_agent)
    assert "100 -> 40 tokens" in compact.lines[0]
    assert run("/diff", fake_agent).lines == (
        "Files modified this session (2):",
        "- a.py",
        "- b.py",
    )

    saved_result = run('/save "named snapshot"', fake_agent)
    assert saved_result.lines[0] == "Session saved: named snapshot"
    assert fake_agent.calls[-1] == ("save_session_snapshot", "named snapshot")

    reset = run("/reset", fake_agent)
    assert reset.lines == ("Conversation and Working Memory reset.",)
    assert fake_agent.calls[-1] == ("reset",)
    assert "Commands:" in run("/help", fake_agent).lines[0]


def test_reset_rejects_arguments_without_changing_state(fake_agent):
    messages = list(fake_agent.messages)
    memory = fake_agent.session_state.working_memory

    result = run("/reset typo", fake_agent)

    assert result.lines == ("Usage: /reset",)
    assert fake_agent.messages == messages
    assert fake_agent.session_state.working_memory is memory
    assert ("reset",) not in fake_agent.calls


def test_unknown_and_invalid_syntax_never_reach_model(fake_agent):
    assert run("/unknown value", fake_agent).lines == (
        "Unknown command: /unknown (try /help)",
    )
    assert run('/save "unterminated', fake_agent).lines[0].startswith(
        "Invalid command syntax:"
    )


def test_state_read_errors_are_rendered_without_terminal_or_provider_calls(fake_agent):
    def fail_resume(_session_id):
        raise ValueError("invalid state identifier")

    def fail_trace(_run_id, _limit):
        raise OSError("trace unavailable")

    fake_agent.resume_session = fail_resume
    fake_agent.recent_trace = fail_trace

    assert run("/session resume ../bad", fake_agent).lines == (
        "Cannot resume ../bad: invalid state identifier.",
    )
    assert run("/trace run-bad", fake_agent).lines == (
        "Cannot read trace: trace unavailable.",
    )


class NoCallFakeLLM:
    def __init__(self, model="fake-model"):
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, *_args, **_kwargs):
        raise AssertionError("CLI state commands must not call a provider")


def test_agent_memory_and_session_command_apis_are_durable(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=NoCallFakeLLM(),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )
    old_id = agent.session_state.session_id
    agent.messages.append({"role": "user", "content": "persist me"})
    agent.session_state.working_memory.current_request = "old task"

    agent.clear_working_memory()

    assert agent.messages == [{"role": "user", "content": "persist me"}]
    persisted = store.load_session(old_id)
    assert persisted is not None
    assert persisted.messages == agent.messages
    assert persisted.working_memory.current_request == ""

    new_state = agent.new_session()
    assert new_state.session_id != old_id
    assert new_state.messages == []
    assert store.load_session(old_id) is not None
    assert store.load_session(new_state.session_id) is not None

    recovery = agent.resume_session(old_id)
    assert recovery is not None
    assert recovery.can_resume is True
    assert agent.session_state.session_id == old_id
    assert agent.messages[0] == {"role": "user", "content": "persist me"}


def test_named_snapshot_preserves_complete_session_and_checkpoint(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=NoCallFakeLLM(),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )
    agent.messages.append({"role": "user", "content": "snapshot me"})
    agent.session_state.working_memory.current_request = "keep memory"
    agent.set_permission_mode("auto")
    original = SessionState.from_dict(agent.session_state.to_dict())

    snapshot_id = agent.save_session_snapshot("named snapshot")

    assert snapshot_id == "named-snapshot"
    snapshot = store.load_session(snapshot_id)
    assert snapshot is not None
    assert snapshot.messages == original.messages
    assert snapshot.working_memory == original.working_memory
    assert snapshot.run_ids == original.run_ids
    assert snapshot.last_checkpoint_id != original.last_checkpoint_id
    checkpoint = store.load_checkpoint(snapshot.last_checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.session_id == snapshot_id
    assert checkpoint.parent_checkpoint_id == original.last_checkpoint_id


def test_diff_paths_are_scoped_to_the_active_session(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=NoCallFakeLLM(),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )
    old_session_id = agent.session_state.session_id
    old_run = RunState(session_id=old_session_id, status="completed")
    agent.session_state.run_ids.append(old_run.run_id)
    store.save_run(old_run)
    store.save_report(Report(
        run_id=old_run.run_id,
        session_id=old_session_id,
        affected_paths=[str(tmp_path / "old.py")],
    ))

    assert agent.modified_paths() == [str(tmp_path / "old.py")]
    agent.new_session()
    assert agent.modified_paths() == []


def test_agent_model_permission_runs_and_queries_use_project_store(tmp_path):
    store = ProjectStore(state_root=tmp_path / "state")
    agent = Agent(
        llm=NoCallFakeLLM(),
        tools=[],
        workspace=WorkspaceContext(tmp_path),
        permission_policy=PermissionPolicy("ask"),
        store=store,
    )

    agent.set_model("next-model")
    model_run_id = agent.session_state.run_ids[-1]
    model_events = store.read_trace(model_run_id).events
    assert any(
        event.event == "run_finished"
        and event.data["previous_model"] == "fake-model"
        and event.data["model"] == "next-model"
        for event in model_events
    )
    model_checkpoint = store.load_checkpoint(agent.session_state.last_checkpoint_id)
    assert model_checkpoint is not None
    assert model_checkpoint.runtime_identity["model"] == "next-model"

    agent.set_permission_mode("read-only")
    permission_run_id = agent.session_state.run_ids[-1]
    permission_events = store.read_trace(permission_run_id).events
    assert any(
        event.event == "run_finished"
        and event.data["previous_permission_mode"] == "ask"
        and event.data["permission_mode"] == "read-only"
        for event in permission_events
    )
    permission_checkpoint = store.load_checkpoint(
        agent.session_state.last_checkpoint_id
    )
    assert permission_checkpoint is not None
    assert permission_checkpoint.runtime_identity["permission_mode"] == "read-only"
    assert agent.tool_executor.permission_policy.mode == "read-only"

    runs = agent.recent_runs()
    assert [run.run_id for run in runs] == [permission_run_id, model_run_id]
    events, warnings = agent.recent_trace(limit=2)
    assert warnings == []
    assert events[-1].event == "run_finished"

    prompt_tokens, completion_tokens, cost = agent.session_token_usage()
    assert (prompt_tokens, completion_tokens) == (0, 0)
    assert cost == 0.0
