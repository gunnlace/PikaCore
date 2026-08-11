"""Repository discovery and canonical path boundaries."""

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorkspaceBoundaryError(ValueError):
    """Raised when a requested path escapes the active workspace."""


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """A lightweight fingerprint of the repository's observable changes."""

    path_fingerprints: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class WorkspaceContext:
    repo_root: Path
    branch: str | None = None
    status: str = ""
    head_commit: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "repo_root", self.repo_root.expanduser().resolve())

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "WorkspaceContext":
        """Discover the containing Git repository without traversing its files."""
        start_path = Path(start or Path.cwd()).expanduser().resolve()
        if start_path.is_file():
            start_path = start_path.parent

        root_result = _run_git(start_path, "rev-parse", "--show-toplevel")
        if root_result is None:
            return cls(repo_root=start_path)

        repo_root = Path(root_result).resolve()
        return cls(
            repo_root=repo_root,
            branch=_run_git(repo_root, "branch", "--show-current") or None,
            status=_run_git(repo_root, "status", "--short") or "",
            head_commit=_run_git(repo_root, "rev-parse", "HEAD") or None,
        )

    def resolve_path(self, user_path: str | Path, *, for_write: bool = False) -> Path:
        """Resolve a user path inside the repository, rejecting every escape."""
        requested = Path(user_path).expanduser()
        candidate = requested if requested.is_absolute() else self.repo_root / requested
        resolved = self._resolve_write_target(candidate) if for_write else candidate.resolve(strict=False)

        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise WorkspaceBoundaryError(f"Path escapes workspace: {user_path}") from exc
        if for_write and resolved == self.repo_root:
            raise WorkspaceBoundaryError("Workspace root cannot be used as a file")
        return resolved

    def snapshot(self) -> WorkspaceSnapshot | None:
        """Capture tracked, staged, and untracked workspace state without mutation."""
        status = _run_git(
            self.repo_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        if status is None:
            return None

        entries = _porcelain_entries(status)
        fingerprints = []
        for relative_path, state in sorted(entries.items()):
            digest = hashlib.sha256(state.encode("ascii"))
            _update_path_digest(digest, self.repo_root / relative_path)
            staged_diff = _run_git(
                self.repo_root,
                "diff",
                "--cached",
                "--binary",
                "--no-ext-diff",
                "--",
                relative_path,
            )
            if staged_diff:
                digest.update(staged_diff.encode("utf-8", errors="surrogateescape"))
            fingerprints.append((relative_path, digest.hexdigest()))
        return WorkspaceSnapshot(tuple(fingerprints))

    def current_branch(self) -> str | None:
        """Return the current branch without relying on construction-time state."""
        branch = _run_git(self.repo_root, "branch", "--show-current")
        if branch is None:
            return self.branch
        return branch or None

    def fingerprint_path(self, user_path: str | Path) -> tuple[str, str]:
        """Fingerprint one relevant workspace path without traversing the repo."""
        resolved = self.resolve_path(user_path, for_write=True)
        relative = resolved.relative_to(self.repo_root).as_posix()
        digest = hashlib.sha256()
        _update_path_digest(digest, resolved)
        return relative, digest.hexdigest()

    @staticmethod
    def _resolve_write_target(candidate: Path) -> Path:
        """Resolve existing symlink parents while allowing a missing final path."""
        cursor = candidate
        missing_parts: list[str] = []
        while not cursor.exists() and not cursor.is_symlink():
            missing_parts.append(cursor.name)
            if cursor == cursor.parent:
                break
            cursor = cursor.parent

        resolved = cursor.resolve(strict=False)
        for part in reversed(missing_parts):
            resolved /= part
        return resolved.resolve(strict=False)


def _run_git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\r\n")


def _porcelain_entries(status: str) -> dict[str, str]:
    """Extract path state for both sides of porcelain v1 -z records."""
    records = status.split("\0")
    entries: dict[str, str] = {}
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        code = record[:2]
        entries[record[3:]] = code
        if "R" in code or "C" in code:
            if index < len(records) and records[index]:
                entries[records[index]] = f"{code}:source"
            index += 1
    return entries


def _update_path_digest(digest, path: Path) -> None:
    """Hash changed file content while never following a symlink."""
    try:
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            return
        if not path.exists():
            digest.update(b"missing\0")
            return
        if not path.is_file():
            digest.update(b"non-file\0")
            return
        digest.update(b"file\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        digest.update(f"unreadable:{type(exc).__name__}".encode())
