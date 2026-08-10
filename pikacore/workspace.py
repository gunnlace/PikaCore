"""Repository discovery and canonical path boundaries."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorkspaceBoundaryError(ValueError):
    """Raised when a requested path escapes the active workspace."""


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
    return result.stdout.strip()
