"""Base class for all tools."""

from abc import ABC, abstractmethod

from ..workspace import WorkspaceContext


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

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

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

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
