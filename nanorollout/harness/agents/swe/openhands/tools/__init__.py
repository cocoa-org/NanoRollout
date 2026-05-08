"""OpenHands-specific tool definitions and default tool composition."""

from .base import BaseTool

from .bash import BashTool
from .base import FinishSignal, ToolParameter
from .finish import FinishTool
from .str_replace_editor import EditorTool
from .task_tracker import TaskTrackerTool
from .think import ThinkTool

__all__ = [
    "BashTool",
    "BaseTool",
    "EditorTool",
    "FinishSignal",
    "FinishTool",
    "TaskTrackerTool",
    "ThinkTool",
    "ToolParameter",
    "build_tools",
]


def build_tools(
    workspace_mount_path_in_sandbox: str | None = None,
) -> list[BaseTool]:
    """Build the default OpenHands toolset for CodeAct agents."""
    return [
        BashTool(),
        EditorTool(workspace_mount_path_in_sandbox=workspace_mount_path_in_sandbox),
        TaskTrackerTool(),
        ThinkTool(),
        FinishTool(),
    ]
