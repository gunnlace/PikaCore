"""Atomic JSON, redacted JSONL, and project artifact layout tests."""

import json

import pytest

from pikacore import store as store_module
from pikacore.state import (
    Checkpoint,
    Report,
    RunState,
    SchemaMismatchError,
    SessionState,
    TraceEvent,
)
from pikacore.store import ProjectStore, atomic_write_json, read_json


def test_atomic_json_failure_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"version": "old"})

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_json(path, {"version": "new"})

    assert read_json(path) == {"version": "old"}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_project_store_round_trips_versioned_artifacts(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    session = SessionState(session_id="session-1", repo_root="/repo", model="fake")
    run = RunState(run_id="run-1", session_id=session.session_id, user_request="hello")
    report = Report(run_id=run.run_id, session_id=session.session_id, completed=True)
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint-1",
        session_id=session.session_id,
        run_id=run.run_id,
    )

    store.save_session(session)
    store.save_run(run)
    store.save_report(report)
    store.save_checkpoint(checkpoint)

    assert store.load_session(session.session_id) == session
    assert store.load_run(run.run_id) == run
    assert store.load_report(run.run_id) == report
    assert store.load_checkpoint(checkpoint.checkpoint_id) == checkpoint
    assert store.session_path(session.session_id) == tmp_path / "sessions" / "session-1.json"
    assert store.task_state_path(run.run_id) == tmp_path / "runs" / "run-1" / "task_state.json"
    assert store.report_path(run.run_id) == tmp_path / "runs" / "run-1" / "report.json"
    assert store.checkpoint_path(checkpoint.checkpoint_id) == (
        tmp_path / "checkpoints" / "checkpoint-1.json"
    )


def test_trace_jsonl_is_recursive_redacted_and_string_limited(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    event = TraceEvent(
        seq=1,
        event="tool_requested",
        session_id="session-1",
        run_id="run-1",
        data={
            "api_key": "sk-should-not-survive",
            "prompt_tokens": 42,
            "tokens": "opaque-secret-value",
            "nested": ["Authorization: Bearer token.value", {"text": "x" * 5000}],
        },
    )

    store.append_trace(event)

    raw = store.trace_path(event.run_id).read_text(encoding="utf-8")
    assert "sk-should-not-survive" not in raw
    assert "token.value" not in raw
    loaded = store.read_trace(event.run_id)
    assert loaded.warnings == []
    assert loaded.events[0].data["api_key"] == "[REDACTED]"
    assert loaded.events[0].data["prompt_tokens"] == 42
    assert loaded.events[0].data["tokens"] == "[REDACTED]"
    assert loaded.events[0].data["nested"][0] == "Authorization: Bearer [REDACTED]"
    assert loaded.events[0].data["nested"][1]["text"].endswith("... [truncated]")


def test_session_json_redacts_secrets_without_mutating_live_state(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    state = SessionState(
        session_id="session-1",
        messages=[
            {"role": "user", "content": "Authorization: Bearer live-token"},
            {"role": "tool", "content": "sk-sensitive-value"},
        ],
    )

    store.save_session(state)

    persisted = store.load_session(state.session_id)
    assert persisted is not None
    assert persisted.messages[0]["content"] == "Authorization: Bearer [REDACTED]"
    assert persisted.messages[1]["content"] == "[REDACTED]"
    assert state.messages[0]["content"] == "Authorization: Bearer live-token"


def test_session_json_redacts_without_truncating_long_messages(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    long_message = "x" * 5000
    state = SessionState(
        session_id="session-long",
        messages=[
            {"role": "user", "content": long_message},
            {"role": "tool", "content": "Bearer sensitive-token"},
        ],
    )

    store.save_session(state)

    persisted = store.load_session(state.session_id)
    assert persisted is not None
    assert persisted.messages[0]["content"] == long_message
    assert persisted.messages[1]["content"] == "Bearer [REDACTED]"


def test_checkpoint_redacts_pending_secrets_without_truncating_arguments(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint-secret",
        pending_tool_calls=[{
            "id": "call-1",
            "name": "write_file",
            "arguments": {
                "api_key": "sk-sensitive-value",
                "content": "x" * 5000,
            },
        }],
    )

    store.save_checkpoint(checkpoint)

    persisted = store.load_checkpoint(checkpoint.checkpoint_id)
    assert persisted is not None
    arguments = persisted.pending_tool_calls[0]["arguments"]
    assert arguments["api_key"] == "[REDACTED]"
    assert arguments["content"] == "x" * 5000


def test_checkpoint_preserves_freshness_hashes_for_secret_like_paths(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint-paths",
        file_freshness={
            "tokenizer.py": "a" * 64,
            "src/api_keys.py": "b" * 64,
            "credentials.txt": "c" * 64,
        },
    )

    store.save_checkpoint(checkpoint)

    persisted = store.load_checkpoint(checkpoint.checkpoint_id)
    assert persisted is not None
    assert persisted.file_freshness == checkpoint.file_freshness


def test_trace_reader_ignores_only_a_corrupt_final_line(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    event = TraceEvent(seq=1, session_id="session-1", run_id="run-1")
    store.append_trace(event)
    with store.trace_path(event.run_id).open("a", encoding="utf-8") as handle:
        handle.write('{"incomplete":')

    result = store.read_trace(event.run_id)

    assert result.events == [event]
    assert len(result.warnings) == 1
    assert "corrupt final trace line" in result.warnings[0]


def test_trace_reader_rejects_corruption_before_the_final_line(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    path = store.trace_path("run-1")
    path.parent.mkdir(parents=True)
    path.write_text('{"broken":\n{}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt trace line 1"):
        store.read_trace("run-1")


def test_trace_reader_does_not_treat_unknown_schema_as_truncation(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    path = store.trace_path("run-1")
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":2}\n', encoding="utf-8")

    with pytest.raises(SchemaMismatchError):
        store.read_trace("run-1")


def test_store_rejects_identifier_path_traversal(tmp_path):
    store = ProjectStore(state_root=tmp_path)

    with pytest.raises(ValueError, match="Invalid state identifier"):
        store.session_path("../outside")
    with pytest.raises(ValueError, match="Invalid state identifier"):
        store.run_dir("/absolute")
    with pytest.raises(ValueError, match="Invalid state identifier"):
        store.checkpoint_path("../outside")


def test_store_load_propagates_schema_mismatch(tmp_path):
    store = ProjectStore(state_root=tmp_path)
    path = store.session_path("future")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    with pytest.raises(SchemaMismatchError):
        store.load_session("future")
