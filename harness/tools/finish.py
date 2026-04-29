"""
Finish tool implementation.
"""

from .base import BaseTool, FinishSignal, ToolParameter, ToolResult

_FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information

The message should include:
- A clear summary of actions taken and their results
- Any next steps for the user
- Explanation if you're unable to complete the task
- Any follow-up questions if more information is needed
"""


class FinishTool(BaseTool):
    """Tool for signaling task completion."""

    @property
    def name(self) -> str:
        return "finish"

    def get_description(self, **kwargs) -> str:
        return _FINISH_DESCRIPTION

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                type="string",
                description="Final message to send to the user",
                required=True,
            ),
        ]

    def execute(
        self,
        environment,
        message: str = "Task completed",
        **kwargs,
    ) -> ToolResult:
        """
        Signal task completion.

        This method raises FinishSignal which should be caught by the agent
        to cleanly exit the execution loop.
        """
        raise FinishSignal(message)
