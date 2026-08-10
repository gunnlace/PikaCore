"""Permission decisions for read-only and side-effecting tools."""

from dataclasses import dataclass
from typing import Literal

from .tools.base import Tool

PermissionMode = Literal["read-only", "ask", "auto"]
PermissionDecision = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class PermissionPolicy:
    mode: PermissionMode = "ask"

    def __post_init__(self):
        if self.mode not in {"read-only", "ask", "auto"}:
            raise ValueError(f"Unknown permission mode: {self.mode}")

    def decide(self, tool: Tool) -> PermissionDecision:
        if tool.read_only:
            return "allow"
        if self.mode == "read-only":
            return "deny"
        if self.mode == "ask":
            return "ask"
        return "allow"
