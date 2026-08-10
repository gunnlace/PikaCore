"""Tool registry."""

from .bash import BashTool
from .read import ReadFileTool
from .write import WriteFileTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .agent import AgentTool


TOOL_TYPES = [
    BashTool,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    AgentTool,
]


def create_tools(workspace=None):
    """Create an independent built-in tool set for one Agent."""
    return [tool_type(workspace=workspace) for tool_type in TOOL_TYPES]


ALL_TOOLS = create_tools()


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
