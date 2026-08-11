"""Structured Working Memory behavior, using only local fakes."""

import copy
from dataclasses import fields

from pikacore.agent import Agent
from pikacore.checkpoint import RecoveryResult
from pikacore.llm import LLMResponse, ToolCall
from pikacore.state import WorkingMemory
from pikacore.store import ProjectStore
from pikacore.tool_executor import ToolExecutionResult
from pikacore.tools.read import ReadFileTool
from pikacore.workspace import WorkspaceContext
from pikacore.working_memory import (
    CheckpointMemoryEvent,
    RecoveryMemoryEvent,
    RunMemoryEvent,
    ToolMemoryEvent,
    UserMemoryEvent,
    WorkingMemoryManager,
    render_working_memory,
)


class FakeLLM:
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(copy.deepcopy(messages))
        return self.responses.pop(0)


def _result(
    tool_name,
    *,
    status="ok",
    content="ok",
    read_paths=None,
    affected_paths=None,
    exit_code=None,
    error_code=None,
    approval="not_required",
):
    return ToolExecutionResult(
        tool_call_id="call",
        tool_name=tool_name,
        content=content,
        status=status,
        read_paths=read_paths or [],
        affected_paths=affected_paths or [],
        exit_code=exit_code,
        error_code=error_code,
        approval=approval,
    )


def _tool_event(tool_name, result, *, arguments=None, index=0):
    return ToolMemoryEvent(
        tool_name=tool_name,
        arguments=arguments or {},
        result=result,
        run_id="run-1",
        occurred_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def test_user_event_replaces_request_and_renderer_omits_unabridged_request(tmp_path):
    memory = WorkingMemory()
    manager = WorkingMemoryManager(memory, WorkspaceContext(tmp_path))
    request = "implement this " + "x" * 500

    assert manager.apply(UserMemoryEvent(request, "run-1", "now"))
    rendered = render_working_memory(memory)

    assert memory.current_request == request
    assert len(memory.task_summary) == 400
    assert request not in rendered
    assert "Task summary:" in rendered


def test_read_then_edit_marks_old_summary_stale_until_structured_reread(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    memory = WorkingMemory()
    manager = WorkingMemoryManager(memory, WorkspaceContext(tmp_path))

    manager.apply(_tool_event(
        "read_file",
        _result(
            "read_file",
            content="1\tvalue = 1",
            read_paths=[str(source)],
        ),
        arguments={"file_path": "source.py"},
    ))
    original_hash = memory.files[0].content_hash
    source.write_text("value = 2\n", encoding="utf-8")
    manager.apply(_tool_event(
        "edit_file",
        _result("edit_file", affected_paths=[str(source)]),
        arguments={"file_path": "source.py"},
        index=1,
    ))

    assert len(memory.files) == 1
    assert memory.files[0].action == "modified"
    assert memory.files[0].fresh is False
    assert memory.files[0].content_hash != original_hash
    assert memory.next_steps == ["Reread source.py"]

    manager.apply(_tool_event(
        "read_file",
        _result("read_file", content="1\tvalue = 2", read_paths=["source.py"]),
        arguments={"file_path": "source.py"},
        index=2,
    ))
    assert memory.files[0].action == "read"
    assert memory.files[0].fresh is True
    assert memory.next_steps == []


def test_commands_errors_rejections_and_capacities_are_bounded(tmp_path):
    memory = WorkingMemory()
    manager = WorkingMemoryManager(memory, WorkspaceContext(tmp_path))

    for index in range(14):
        manager.apply(_tool_event(
            "bash",
            _result(
                "bash",
                status="error",
                exit_code=index,
                error_code=f"exit-{index}",
            ),
            arguments={"command": f"command-{index}"},
            index=index,
        ))
    manager.apply(_tool_event(
        "write_file",
        _result(
            "write_file",
            status="rejected",
            error_code="approval-rejected",
            approval="rejected",
        ),
        arguments={"file_path": "denied.txt"},
        index=20,
    ))

    assert [item.command for item in memory.recent_commands] == [
        f"command-{index}" for index in range(4, 14)
    ]
    assert len(memory.blockers) == 10
    assert memory.blockers[-1] == "write_file: approval-rejected"


def test_files_and_checkpoint_next_steps_are_bounded_and_deduplicated(tmp_path):
    memory = WorkingMemory()
    manager = WorkingMemoryManager(memory, WorkspaceContext(tmp_path))
    for index in range(35):
        path = tmp_path / f"file-{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        manager.apply(_tool_event(
            "read_file",
            _result("read_file", content=str(index), read_paths=[str(path)]),
            arguments={"file_path": str(path)},
            index=index % 60,
        ))
    for index in range(13):
        manager.apply(CheckpointMemoryEvent(
            file_freshness={},
            pending_tool_names=["write_file"],
            next_suggested_action=f"check-{index}",
            occurred_at=f"checkpoint-{index}",
        ))

    assert len(memory.files) == 30
    assert memory.files[0].path == "file-5.txt"
    assert len(memory.next_steps) == 10
    assert memory.next_steps[-1] == "check-12"


def test_recovery_events_mark_freshness_blockers_and_deterministic_checks(tmp_path):
    path = tmp_path / "tracked.py"
    path.write_text("old", encoding="utf-8")
    memory = WorkingMemory()
    manager = WorkingMemoryManager(memory, WorkspaceContext(tmp_path))
    manager.apply(_tool_event(
        "read_file",
        _result("read_file", content="old", read_paths=["tracked.py"]),
        arguments={"file_path": "tracked.py"},
    ))

    manager.apply(RecoveryMemoryEvent(
        result=RecoveryResult(status="files-stale", stale_paths=["tracked.py"]),
        occurred_at="recovery",
    ))
    manager.apply(RecoveryMemoryEvent(
        result=RecoveryResult(
            status="incomplete-tool-call",
            pending_tool_names=["bash"],
            requires_workspace_inspection=True,
        ),
        occurred_at="recovery-2",
    ))

    assert memory.files[0].fresh is False
    assert "Reread tracked.py" in memory.next_steps
    assert any("Inspect workspace" in item for item in memory.next_steps)
    assert any("stale files" in item for item in memory.blockers)
    assert any("incomplete tool calls" in item for item in memory.blockers)


def test_run_event_has_no_final_answer_channel_or_phrase_parser(tmp_path):
    memory = WorkingMemory(next_steps=["Keep structured check"])
    manager = WorkingMemoryManager(memory, WorkspaceContext(tmp_path))

    changed = manager.apply(RunMemoryEvent(
        status="completed",
        stop_reason="completed",
        run_id="run-1",
        occurred_at="finished",
    ))

    assert changed is False
    assert memory.next_steps == ["Keep structured check"]
    assert "final_answer" not in {item.name for item in fields(RunMemoryEvent)}


def test_agent_persists_and_prompts_with_working_memory_without_api_calls(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("remembered contents\n", encoding="utf-8")
    store = ProjectStore(state_root=tmp_path / "state")
    llm = FakeLLM([
        LLMResponse(tool_calls=[
            ToolCall("read-1", "read_file", {"file_path": "sample.txt"})
        ]),
        LLMResponse(content="Next steps: this sentence must not update memory."),
    ])
    agent = Agent(
        llm=llm,
        tools=[ReadFileTool()],
        workspace=WorkspaceContext(tmp_path),
        store=store,
    )

    agent.chat("Inspect sample.txt")
    restored = store.load_session(agent.session_state.session_id)

    assert restored is not None
    assert restored.working_memory.current_request == "Inspect sample.txt"
    assert restored.working_memory.files[0].path == "sample.txt"
    assert restored.working_memory.files[0].fresh is True
    assert not any("this sentence" in item for item in restored.working_memory.next_steps)
    assert "[Working memory]" in llm.calls[0][1]["content"]
    assert "sample.txt (read, fresh)" in llm.calls[1][1]["content"]
    assert not (store.state_root / "memory").exists()
