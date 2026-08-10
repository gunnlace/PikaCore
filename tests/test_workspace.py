"""Workspace discovery and filesystem boundary tests."""

import subprocess

import pytest

from pikacore.tools.read import ReadFileTool
from pikacore.tools.write import WriteFileTool
from pikacore.workspace import WorkspaceBoundaryError, WorkspaceContext


def _symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_discover_uses_git_toplevel_from_nested_directory(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "package"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "untracked.txt").write_text("status\n", encoding="utf-8")

    workspace = WorkspaceContext.discover(nested)

    assert workspace.repo_root == repo.resolve()
    assert "untracked.txt" in workspace.status


def test_discover_falls_back_to_start_outside_git(tmp_path):
    start = tmp_path / "plain"
    start.mkdir()

    workspace = WorkspaceContext.discover(start)

    assert workspace.repo_root == start.resolve()
    assert workspace.branch is None
    assert workspace.status == ""
    assert workspace.head_commit is None


def test_relative_paths_are_rooted_at_repo_not_process_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    repo.mkdir()
    elsewhere.mkdir()
    target = repo / "inside.txt"
    target.write_text("inside\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)

    workspace = WorkspaceContext(repo)

    assert workspace.resolve_path("inside.txt") == target.resolve()
    assert workspace.resolve_path(target) == target.resolve()


@pytest.mark.parametrize("requested", ["../outside.txt", "nested/../../outside.txt"])
def test_parent_traversal_is_rejected(tmp_path, requested):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(WorkspaceBoundaryError, match="escapes workspace"):
        WorkspaceContext(repo).resolve_path(requested)


def test_absolute_path_outside_repo_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside.txt"
    repo.mkdir()
    outside.write_text("secret\n", encoding="utf-8")

    with pytest.raises(WorkspaceBoundaryError, match="escapes workspace"):
        WorkspaceContext(repo).resolve_path(outside)


def test_read_rejects_symlink_escape(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside.txt"
    repo.mkdir()
    outside.write_text("secret\n", encoding="utf-8")
    link = repo / "linked.txt"
    _symlink(link, outside)

    result = ReadFileTool(workspace=WorkspaceContext(repo)).execute("linked.txt")

    assert "escapes workspace" in result
    assert "secret" not in result


def test_write_allows_missing_target_below_existing_repo_parent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tool = WriteFileTool(workspace=WorkspaceContext(repo))

    result = tool.execute("new/deep/file.txt", "created\n")

    assert result.startswith("Wrote")
    assert (repo / "new" / "deep" / "file.txt").read_text(encoding="utf-8") == "created\n"


def test_write_rejects_missing_target_below_symlink_parent(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    link = repo / "linked-dir"
    _symlink(link, outside, directory=True)
    tool = WriteFileTool(workspace=WorkspaceContext(repo))

    result = tool.execute("linked-dir/new/file.txt", "blocked\n")

    assert "escapes workspace" in result
    assert not (outside / "new" / "file.txt").exists()


def test_write_rejects_repo_root_as_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = WriteFileTool(workspace=WorkspaceContext(repo)).execute(".", "blocked\n")

    assert "Workspace root cannot be used as a file" in result
