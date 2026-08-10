"""Base class for all tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..workspace import WorkspaceContext


@dataclass(frozen=True)
class ToolOutput:
    """Execution metadata kept separate from the model-facing text."""

    content: str
    status: Literal["ok", "error", "partial"] = "ok"
    error_code: str | None = None
    exit_code: int | None = None
    output_truncated: bool = False


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args
    risk_level: Literal["low", "medium", "high"] = "low"
    read_only: bool = True
    path_parameters: dict[str, bool] = {}

    def __init__(self, workspace: WorkspaceContext | None = None):
        self.workspace = workspace

    def bind_workspace(self, workspace: WorkspaceContext) -> "Tool":
        self.workspace = workspace
        return self

    @property
    def workspace_context(self) -> WorkspaceContext:
        return self.workspace or WorkspaceContext.discover()

    def resolve_path(self, user_path: str, *, for_write: bool = False):
        return self.workspace_context.resolve_path(user_path, for_write=for_write)

    def normalize_arguments(self, arguments: dict) -> dict:
        normalized = dict(arguments)
        for name, for_write in self.path_parameters.items():
            if name in normalized:
                normalized[name] = str(self.resolve_path(normalized[name], for_write=for_write))
        return normalized

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

    def execute_structured(self, **kwargs) -> ToolOutput:
        """Execute with metadata; legacy tools retain their text contract."""
        content = self.execute(**kwargs)
        is_error = content.startswith(("Error", "Invalid regex", "⚠ Blocked"))
        return ToolOutput(
            content=content,
            status="error" if is_error else "ok",
            error_code="tool-error" if is_error else None,
            output_truncated="truncated" in content.lower(),
        )

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
