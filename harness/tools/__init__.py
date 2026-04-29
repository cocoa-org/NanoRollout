"""
Tool implementations for OH-Core.
"""

from .base import BaseTool, FinishSignal, ToolParameter, ToolResult
from .bash import BashTool
from .finish import FinishTool
from .str_replace_editor import EditorTool
from .task_tracker import TaskTrackerTool
from .think import ThinkTool

__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "FinishSignal",
    "BashTool",
    "FinishTool",
    "EditorTool",
    "TaskTrackerTool",
    "ThinkTool",
    "get_default_tools",
]


def get_default_tools(
    workspace_mount_path_in_sandbox: str | None = None,
) -> list[BaseTool]:
    """Get the default set of tools for the CodeAct agent."""
    return [
        BashTool(),
        EditorTool(workspace_mount_path_in_sandbox=workspace_mount_path_in_sandbox),
        TaskTrackerTool(),
        ThinkTool(),
        FinishTool(),
    ]
