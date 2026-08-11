"""File pattern matching."""

from pathlib import Path
from .base import Tool, ToolOutput


class GlobTool(Tool):
    name = "glob"
    path_parameters = {"path": False}
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> str:
        return self._glob(pattern, path)[0]

    def execute_structured(self, pattern: str, path: str = ".") -> ToolOutput:
        content, read_paths, review_paths = self._glob(pattern, path)
        is_error = content.startswith("Error")
        return ToolOutput(
            content=content,
            status="error" if is_error else "ok",
            error_code="tool-error" if is_error else None,
            read_paths=tuple(str(path) for path in read_paths),
            freshness_review_paths=tuple(str(path) for path in review_paths),
        )

    def _glob(self, pattern: str, path: str):
        try:
            pattern_path = Path(pattern)
            if pattern_path.is_absolute() or ".." in pattern_path.parts:
                return f"Error: glob pattern escapes workspace: {pattern}", [], []
            base = self.resolve_path(path)
            if not base.is_dir():
                return f"Error: {path} is not a directory", [], []

            hits = []
            for hit in base.glob(pattern):
                try:
                    hits.append(self.resolve_path(str(hit)))
                except ValueError:
                    continue
            # sort by mtime, newest first
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            read_paths = [hit for hit in shown if hit.is_file()]
            return result or "No files matched.", read_paths, [base]
        except Exception as e:
            return f"Error: {e}", [], []
