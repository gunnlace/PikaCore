"""Tests for core modules: config, context, session, imports."""

import sys

import pytest

from pikacore import Agent, LLM, Config, ALL_TOOLS, __version__
from pikacore import session as session_module
from pikacore.cli import _parse_args
from pikacore.context import ContextManager, estimate_tokens
from pikacore.session import save_session, load_session, list_sessions
from pikacore.tools import get_tool


def test_version():
    assert __version__ == "0.1.0"


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    assert len(ALL_TOOLS) == 7


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("PIKACORE_MODEL", "test-model")
    c = Config.from_env()
    assert c.model == "test-model"


def test_config_prefers_pikacore_env_and_supports_corecoder_fallback(monkeypatch):
    monkeypatch.setenv("PIKACORE_MODEL", "new-model")
    monkeypatch.setenv("CORECODER_MODEL", "legacy-model")
    assert Config.from_env().model == "new-model"

    monkeypatch.delenv("PIKACORE_MODEL")
    assert Config.from_env().model == "legacy-model"


def test_config_keeps_openai_env_and_legacy_api_key_is_last_fallback(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("PIKACORE_API_KEY", "new-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("CORECODER_API_KEY", "legacy-key")
    assert Config.from_env().api_key == "new-key"

    monkeypatch.delenv("PIKACORE_API_KEY")
    assert Config.from_env().api_key == "openai-key"

    monkeypatch.delenv("OPENAI_API_KEY")
    assert Config.from_env().api_key == "legacy-key"


@pytest.mark.parametrize(
    ("primary", "legacy", "attribute", "primary_value", "primary_expected", "legacy_expected"),
    [
        ("PIKACORE_MAX_TOKENS", "CORECODER_MAX_TOKENS", "max_tokens", "8192", 8192, 2048),
        ("PIKACORE_TEMPERATURE", "CORECODER_TEMPERATURE", "temperature", "0.5", 0.5, 0.25),
        ("PIKACORE_MAX_CONTEXT", "CORECODER_MAX_CONTEXT", "max_context_tokens", "64000", 64000, 32000),
        ("PIKACORE_PROVIDER", "CORECODER_PROVIDER", "provider", "litellm", "litellm", "openai"),
    ],
)
def test_config_uses_legacy_fallback_only_when_primary_is_absent(
    monkeypatch, primary, legacy, attribute, primary_value, primary_expected, legacy_expected
):
    monkeypatch.setenv(primary, primary_value)
    monkeypatch.setenv(legacy, str(legacy_expected))
    assert getattr(Config.from_env(), attribute) == primary_expected

    monkeypatch.delenv(primary)
    assert getattr(Config.from_env(), attribute) == legacy_expected


def test_config_base_url_priority(monkeypatch):
    monkeypatch.setenv("PIKACORE_BASE_URL", "https://pikacore.example")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example")
    monkeypatch.setenv("CORECODER_BASE_URL", "https://corecoder.example")
    assert Config.from_env().base_url == "https://pikacore.example"

    monkeypatch.delenv("PIKACORE_BASE_URL")
    assert Config.from_env().base_url == "https://openai.example"

    monkeypatch.delenv("OPENAI_BASE_URL")
    assert Config.from_env().base_url == "https://corecoder.example"


def test_config_defaults(monkeypatch):
    # clear relevant env vars without leaking the change into other tests
    monkeypatch.delenv("PIKACORE_MODEL", raising=False)
    monkeypatch.delenv("PIKACORE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("CORECODER_MODEL", raising=False)
    monkeypatch.delenv("CORECODER_MAX_TOKENS", raising=False)

    c = Config.from_env()
    assert c.model == "gpt-5.5"
    assert c.max_tokens == 4096
    assert c.temperature == 0.0


def test_cli_help_uses_pikacore_names_and_marks_legacy_fallback(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pikacore", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        _parse_args()

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "usage: pikacore" in help_text
    assert "$PIKACORE_MODEL" in help_text
    assert "$CORECODER_MODEL is a compatibility fallback" in help_text
    assert "$PIKACORE_BASE_URL or $OPENAI_BASE_URL" in help_text
    assert "$CORECODER_BASE_URL is a compatibility fallback" in help_text
    assert "$PIKACORE_API_KEY or $OPENAI_API_KEY" in help_text
    assert "$CORECODER_API_KEY is a compatibility fallback" in help_text


# --- Context ---

def test_estimate_tokens():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 100


def test_context_snip():
    ctx = ContextManager(max_tokens=3000)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": "x\n" * 1000},
    ]
    before = estimate_tokens(msgs)
    ctx._snip_tool_outputs(msgs)
    after = estimate_tokens(msgs)
    assert after < before


def test_context_compress():
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 2000})
    before = estimate_tokens(msgs)
    ctx.maybe_compress(msgs, None)
    after = estimate_tokens(msgs)
    assert after < before
    assert len(msgs) < 40  # should be compressed


def test_safe_split_never_orphans_a_tool_message():
    """The kept tail must not begin with a 'tool' message - it would be severed
    from the assistant tool_calls that produced it, which the API rejects."""
    ctx = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "tool", "tool_call_id": "c2", "content": "result2"},
    ]
    split = ctx._safe_split(messages, keep_recent=1)
    assert messages[split].get("role") != "tool"


def test_compress_never_leaves_an_orphan_tool_reply():
    """After summarisation every tool reply must still follow its tool_calls."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "b" * 800})
    ctx.maybe_compress(msgs, None)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            prev = msgs[i - 1]
            assert prev.get("role") == "tool" or prev.get("tool_calls"), f"orphan tool at {i}"


# --- Session ---

def test_session_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    save_session(msgs, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == msgs
    assert loaded[1] == "test-model"


def test_session_name_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "../Research Notes!")

    assert sid == "Research-Notes"
    assert (tmp_path / "Research-Notes.json").exists()
    assert load_session("../Research Notes!") is not None


def test_session_not_found():
    assert load_session("nonexistent_session_id") is None


def test_list_sessions():
    sessions = list_sessions()
    assert isinstance(sessions, list)


# --- Cost estimation ---

def test_cost_estimation_known_model():
    from pikacore.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "gpt-5.4"
    llm.total_prompt_tokens = 1_000_000
    llm.total_completion_tokens = 500_000
    cost = llm.estimated_cost
    assert cost is not None
    assert cost == 2.5 + 7.5  # $2.5/M in + $15/M out * 0.5M

def test_cost_estimation_unknown_model():
    from pikacore.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "some-custom-model"
    llm.total_prompt_tokens = 1000
    llm.total_completion_tokens = 500
    assert llm.estimated_cost is None


# --- Changed files tracking ---

def test_edit_tracks_changed_files(tmp_path):
    from pikacore.tools.edit import _changed_files
    _changed_files.clear()
    edit = get_tool("edit_file")
    path = tmp_path / "sample.py"
    path.write_text("aaa\nbbb\n")
    edit.execute(file_path=str(path), old_string="aaa", new_string="zzz")
    assert any(str(path) in p for p in _changed_files)
    _changed_files.clear()


def test_write_tracks_changed_files(tmp_path):
    from pikacore.tools.edit import _changed_files
    _changed_files.clear()
    write = get_tool("write_file")
    path = tmp_path / "tracked.txt"
    write.execute(file_path=str(path), content="tracked\n")
    assert any(path.name in p for p in _changed_files)
    _changed_files.clear()


# --- Agent tool execution ---

def test_agent_tool_scope_is_per_instance():
    """An Agent restricted to a subset of tools must not resolve tools outside it."""
    only_read = [get_tool("read_file")]
    agent = Agent(llm=LLM.__new__(LLM), tools=only_read)
    assert set(agent._tool_by_name) == {"read_file"}

    class _TC:
        name = "bash"  # a real, registered tool - but not in this agent's set
        id = "x"
        arguments = {"command": "echo hi"}

    assert "unknown tool 'bash'" in agent._exec_tool(_TC())


def test_exec_tool_distinguishes_bad_args_from_internal_error():
    """A TypeError raised inside a tool must not be reported as bad arguments."""
    from pikacore.tools.base import Tool

    class _Boom(Tool):
        name = "boom"
        description = "raises TypeError internally"
        parameters = {"type": "object", "properties": {}, "required": []}

        def execute(self):
            raise TypeError("internal explosion")

    agent = Agent(llm=LLM.__new__(LLM), tools=[_Boom()])

    class _BadArgs:
        name, id, arguments = "boom", "1", {"unexpected": 1}

    class _Good:
        name, id, arguments = "boom", "2", {}

    assert "bad arguments" in agent._exec_tool(_BadArgs())
    assert "Error executing boom" in agent._exec_tool(_Good())
    assert "bad arguments" not in agent._exec_tool(_Good())


def test_interrupt_backfills_missing_tool_replies():
    """A half-finished tool round must be repaired so history stays valid."""
    agent = Agent(llm=LLM.__new__(LLM), tools=[])
    agent.messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
    ]

    class _TC:
        def __init__(self, i):
            self.id = i

    agent._answer_pending_tool_calls([_TC("a"), _TC("b")])
    replies = [m for m in agent.messages if m.get("role") == "tool"]
    ids = [m["tool_call_id"] for m in replies]
    assert sorted(ids) == ["a", "b"]
    assert ids.count("a") == 1  # the already-answered call wasn't duplicated
