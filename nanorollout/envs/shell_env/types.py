"""Shared result types for shell environment tool execution."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of a command execution."""

    output: str
    exit_code: int


@dataclass
class ToolResult:
    """Result of a tool execution."""

    output: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


GIT_USER_EMAIL = "evaluation@openhands.dev"
GIT_USER_NAME = "OpenHands Evaluation"
GIT_COMMIT_MESSAGE = "patch"


DEFAULT_TRUNCATE_NOTICE = "<response clipped>"
DEFAULT_TRUNCATE_NOTICE_WITH_PERSIST = (
    "<response clipped; full output saved to {file_path} at line {line_num}>"
)
TOOL_LOGGER_NAME = "nanorollout.envs.tools"
