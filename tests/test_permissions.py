"""Permission modes, tool risk metadata, and environment filtering."""

import os
import sys

import pytest

from pikacore.permissions import PermissionPolicy
from pikacore.security import redact, sanitize_environment
from pikacore.tools.agent import AgentTool
from pikacore.tools.bash import BashTool
from pikacore.tools.edit import EditFileTool
from pikacore.tools.glob_tool import GlobTool
from pikacore.tools.grep import GrepTool
from pikacore.tools.read import ReadFileTool
from pikacore.tools.write import WriteFileTool
from pikacore.workspace import WorkspaceContext


@pytest.mark.parametrize("mode", ["read-only", "ask", "auto"])
def test_read_only_tools_are_allowed_in_every_mode(mode):
    assert PermissionPolicy(mode).decide(ReadFileTool()) == "allow"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("read-only", "deny"),
        ("ask", "ask"),
        ("auto", "allow"),
    ],
)
@pytest.mark.parametrize("tool", [WriteFileTool(), EditFileTool(), BashTool(), AgentTool()])
def test_mutating_tool_decisions_cover_all_modes(mode, expected, tool):
    assert PermissionPolicy(mode).decide(tool) == expected


def test_default_permission_mode_is_ask():
    assert PermissionPolicy().mode == "ask"
    assert PermissionPolicy().decide(WriteFileTool()) == "ask"


def test_builtin_tool_risk_metadata():
    assert (ReadFileTool().risk_level, ReadFileTool().read_only) == ("low", True)
    assert (GlobTool().risk_level, GlobTool().read_only) == ("low", True)
    assert (GrepTool().risk_level, GrepTool().read_only) == ("low", True)
    assert (WriteFileTool().risk_level, WriteFileTool().read_only) == ("medium", False)
    assert (EditFileTool().risk_level, EditFileTool().read_only) == ("medium", False)
    assert (BashTool().risk_level, BashTool().read_only) == ("high", False)
    assert (AgentTool().risk_level, AgentTool().read_only) == ("high", False)


def test_sanitize_environment_uses_allowlist_and_removes_secrets():
    source = {
        "PATH": "/bin",
        "HOME": "/home/user",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": "/project",
        "ComSpec": r"C:\Windows\System32\cmd.exe",
        "SystemRoot": r"C:\Windows",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "TEMP": r"C:\Temp",
        "USERPROFILE": r"C:\Users\runner",
        "OPENAI_API_KEY": "openai-secret",
        "PIKACORE_TOKEN": "pikacore-secret",
        "DATABASE_PASSWORD": "db-secret",
        "UNRELATED": "drop-me",
    }

    assert sanitize_environment(source) == {
        "PATH": "/bin",
        "HOME": "/home/user",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": "/project",
        "ComSpec": r"C:\Windows\System32\cmd.exe",
        "SystemRoot": r"C:\Windows",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "TEMP": r"C:\Temp",
        "USERPROFILE": r"C:\Users\runner",
    }


def test_bash_subprocess_environment_does_not_contain_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("PIKACORE_API_KEY", "must-not-leak-either")
    bash = BashTool(workspace=WorkspaceContext(tmp_path))
    command = (
        f'"{sys.executable}" -c "import os; '
        "print(os.getenv('OPENAI_API_KEY')); print(os.getenv('PIKACORE_API_KEY'))\""
    )

    result = bash.execute(command)

    assert result.splitlines() == ["None", "None"]


def test_redact_handles_nested_secrets_tokens_urls_and_long_strings():
    source = {
        "api_key": "sk-should-not-survive",
        "nested": [
            "Authorization: Bearer abc.def-123",
            ("https://alice:password@example.com/path", "pk-abcdefghijk"),
        ],
        "long": "x" * 5000,
        "count": 3,
    }

    result = redact(source)

    assert result["api_key"] == "[REDACTED]"
    assert result["nested"][0] == "Authorization: Bearer [REDACTED]"
    assert result["nested"][1][0] == "https://[REDACTED]@example.com/path"
    assert result["nested"][1][1] == "[REDACTED]"
    assert result["long"].endswith("... [truncated]")
    assert len(result["long"]) < 5000
    assert result["count"] == 3


def test_shell_allowlist_preserves_current_path():
    sanitized = sanitize_environment(os.environ)
    if "PATH" in os.environ:
        assert sanitized["PATH"] == os.environ["PATH"]
