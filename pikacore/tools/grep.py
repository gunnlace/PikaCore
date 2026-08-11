"""Content search with regex support."""

import re
from .base import Tool, ToolOutput

# skip these dirs to avoid noise
_SKIP_DIRS = {
    ".git",
    ".pikacore",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
}


class GrepTool(Tool):
    name = "grep"
    path_parameters = {"path": False}
    description = (
        "Search file contents with regex. "
        "Returns matching lines with file path and line number."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: cwd)",
            },
            "include": {
                "type": "string",
                "description": "Only search files matching this glob (e.g. '*.py')",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", include: str | None = None) -> str:
        return self._search(pattern, path, include)[0]

    def execute_structured(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
    ) -> ToolOutput:
        content, read_paths, review_paths = self._search(pattern, path, include)
        is_error = content.startswith(("Error", "Invalid regex"))
        return ToolOutput(
            content=content,
            status="error" if is_error else "ok",
            error_code="tool-error" if is_error else None,
            read_paths=tuple(str(path) for path in read_paths),
            freshness_review_paths=tuple(str(path) for path in review_paths),
        )

    def _search(self, pattern: str, path: str, include: str | None):
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}", [], []

        try:
            base = self.resolve_path(path)
        except ValueError as e:
            return f"Error: {e}", [], []
        if not base.exists():
            return f"Error: {path} not found", [], []

        if base.is_file():
            files = [base]
            review_paths = []
        else:
            files = self._walk(base, include)
            review_paths = [base]

        matches = []
        read_paths = []
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            read_paths.append(fp)
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{fp}:{lineno}: {line.rstrip()}")
                    if len(matches) >= 200:
                        matches.append("... (200 match limit reached)")
                        return "\n".join(matches), read_paths, review_paths

        return (
            "\n".join(matches) if matches else "No matches found.",
            read_paths,
            review_paths,
        )

    def _walk(self, root, include: str | None) -> list:
        """Walk dir tree, skipping junk dirs."""
        results = []
        for item in root.rglob(include or "*"):
            # skip junk dirs *inside* the search root - matching item.parts would
            # also catch an ancestor named e.g. "build" and hide the whole tree
            if any(part in _SKIP_DIRS for part in item.relative_to(root).parts):
                continue
            try:
                safe_item = self.resolve_path(str(item))
            except ValueError:
                continue
            if safe_item.is_file():
                results.append(safe_item)
            if len(results) >= 5000:
                break
        return results
