"""Versioned Phase 3 state model tests."""

import pytest

from pikacore.state import (
    SCHEMA_VERSION,
    Checkpoint,
    Report,
    RunState,
    SchemaMismatchError,
    SessionState,
    TraceEvent,
    WorkingMemory,
)


@pytest.mark.parametrize(
    "state",
    [
        SessionState(repo_root="/repo", model="fake", messages=[{"role": "user"}]),
        RunState(session_id="session-1", user_request="test"),
        Checkpoint(session_id="session-1", run_id="run-1"),
        TraceEvent(seq=1, session_id="session-1", run_id="run-1", data={"ok": True}),
        Report(run_id="run-1", session_id="session-1", completed=True),
    ],
)
def test_state_models_round_trip_through_dict(state):
    restored = type(state).from_dict(state.to_dict())

    assert restored == state
    assert restored.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize(
    "state_type", [SessionState, RunState, Checkpoint, TraceEvent, Report]
)
def test_state_models_fail_closed_on_unknown_schema(state_type):
    with pytest.raises(SchemaMismatchError) as exc_info:
        state_type.from_dict({"schema_version": SCHEMA_VERSION + 1})

    assert exc_info.value.error_code == "schema-mismatch"


def test_session_state_uses_structured_working_memory_and_upgrades_placeholder():
    state = SessionState()

    assert isinstance(state.working_memory, WorkingMemory)
    assert state.working_memory.files == []
    assert state.last_checkpoint_id is None

    data = state.to_dict()
    data["working_memory"] = {}
    restored = SessionState.from_dict(data)

    assert restored.working_memory.current_request == ""
    assert restored.working_memory.files == []
