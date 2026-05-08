"""Shared result types for shell environment tool execution."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result of a tool execution."""

    output: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
