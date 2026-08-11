import json
from types import SimpleNamespace

import pytest

from pikacore import session as session_module
from pikacore.checkpoint import RecoveryResult
from pikacore.session import load_session, save_session
from pikacore.state import SchemaMismatchError, SessionState
from pikacore.store import atomic_write_json


def test_default_session_directory_uses_pikacore_name():
    assert session_module.SESSIONS_DIR == session_module._find_repo_root() / ".pikacore" / "sessions"


def test_find_repo_root_walks_up_from_nested_directory(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "package"
    nested.mkdir(parents=True)
    (repo / ".git").write_text("gitdir: /tmp/example.git\n", encoding="utf-8")

    assert session_module._find_repo_root(nested) == repo


def test_session_directories_are_isolated_by_repository(tmp_path):
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    (repo_a / ".git").mkdir(parents=True)
    (repo_b / ".git").mkdir(parents=True)

    sessions_a = session_module._find_repo_root(repo_a) / ".pikacore" / "sessions"
    sessions_b = session_module._find_repo_root(repo_b) / ".pikacore" / "sessions"

    assert sessions_a != sessions_b


def test_find_repo_root_falls_back_to_start_directory_outside_git(tmp_path):
    start = tmp_path / "plain-directory"
    start.mkdir()

    assert session_module._find_repo_root(start) == start.resolve()


def test_default_session_ids_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    first_id = save_session([{"role": "user", "content": "first"}], "model-a")
    second_id = save_session([{"role": "user", "content": "second"}], "model-b")

    assert first_id != second_id
    first = load_session(first_id)
    second = load_session(second_id)
    assert first is not None
    assert first.session_id == first_id
    assert first.messages == [{"role": "user", "content": "first"}]
    assert first.model == "model-a"
    assert second is not None
    assert second.session_id == second_id
    assert second.messages == [{"role": "user", "content": "second"}]
    assert second.model == "model-b"


def test_session_id_path_traversal_is_neutralized(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", "../../etc/passwd")

    assert sid == "passwd"
    assert (tmp_path / "passwd.json").exists()
    # the same traversal string round-trips through the parent-dir boundary check
    loaded = load_session("../../etc/passwd")
    assert loaded is not None
    assert loaded.session_id == "passwd"
    assert loaded.messages == [{"role": "user", "content": "x"}]
    assert loaded.model == "m"


def test_session_id_absolute_path_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", "/etc/shadow")

    assert sid == "shadow"
    assert (tmp_path / "shadow.json").exists()


def test_session_id_windows_backslash_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", r"..\..\secret")

    assert sid == "secret"


def test_session_id_length_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", "a" * 500)

    assert len(sid) <= 100
    assert (tmp_path / f"{sid}.json").exists()


def test_corrupt_session_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    (tmp_path / "broken.json").write_text("{ not valid json", encoding="utf-8")

    assert load_session("broken") is None


def test_session_roundtrips_unicode(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    msgs = [{"role": "user", "content": "请帮我修复这个 bug"}]
    sid = save_session(msgs, "model-zh")

    raw = (tmp_path / f"{sid}.json").read_bytes()
    assert "请帮我修复这个 bug".encode("utf-8") in raw
    loaded = load_session(sid)
    assert loaded is not None
    assert loaded.messages == msgs
    assert loaded.model == "model-zh"


def test_load_session_preserves_pre_schema_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    (tmp_path / "legacy.json").write_text(
        '{"id":"legacy","model":"old-model","saved_at":"yesterday",'
        '"messages":[{"role":"user","content":"old"}]}',
        encoding="utf-8",
    )

    loaded = load_session("legacy")
    assert loaded is not None
    assert loaded.session_id == "legacy"
    assert loaded.created_at == "yesterday"
    assert loaded.updated_at == "yesterday"
    assert loaded.messages == [{"role": "user", "content": "old"}]
    assert loaded.model == "old-model"
    assert loaded.run_ids == []
    assert session_module.list_sessions()[0]["saved_at"] == "yesterday"


def test_load_session_preserves_complete_versioned_state(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    state = SessionState(
        session_id="resume-me",
        created_at="created",
        updated_at="updated",
        repo_root="/original/repo",
        model="saved-model",
        messages=[{"role": "user", "content": "old context"}],
        run_ids=["run-1", "run-2"],
    )
    atomic_write_json(tmp_path / "resume-me.json", state.to_dict())

    assert load_session("resume-me") == state


def test_load_session_propagates_schema_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    (tmp_path / "future.json").write_text(
        json.dumps({"schema_version": 2}),
        encoding="utf-8",
    )

    with pytest.raises(SchemaMismatchError) as exc_info:
        load_session("future")

    assert exc_info.value.error_code == "schema-mismatch"


def test_cli_resume_constructs_agent_with_complete_session_state(monkeypatch):
    from pikacore import cli
    from pikacore.config import Config

    state = SessionState(
        session_id="resume-me",
        model="saved-model",
        run_ids=["run-earlier"],
    )
    args = SimpleNamespace(
        model=None,
        base_url=None,
        api_key=None,
        prompt="continue",
        resume="resume-me",
        permissions="auto",
    )
    observed = {}

    class FakeLLM:
        def __init__(self, *, model, **_kwargs):
            self.model = model

    class FakeAgent:
        def __init__(self, *, llm, session_state, **_kwargs):
            self.llm = llm
            observed["llm"] = llm
            observed["session_state"] = session_state

        def recover_session(self):
            observed["recovered"] = True
            return RecoveryResult(status="full-valid")

    monkeypatch.setattr(cli, "_parse_args", lambda: args)
    monkeypatch.setattr(cli.Config, "from_env", lambda: Config(api_key="fake-key"))
    monkeypatch.setattr(cli, "load_session", lambda _session_id: state)
    monkeypatch.setattr(cli, "LLM", FakeLLM)
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    monkeypatch.setattr(
        cli,
        "_run_once",
        lambda agent, prompt: observed.update(agent=agent, prompt=prompt),
    )

    cli.main()

    assert observed["session_state"] is state
    assert observed["llm"].model == "saved-model"
    assert observed["recovered"] is True
    assert observed["prompt"] == "continue"
