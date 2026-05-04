"""
CodeAct Agent implementation.

This module provides the CodeAct agent that uses function calling
to interact with an environment.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from brew.harness.agents.swe.base import AgentConfig, AgentResult, BaseAgent
from brew.harness.tools import get_default_tools
from brew.harness.tools.base import BaseTool, FinishSignal

from .prompts import get_core_system_prompt

if TYPE_CHECKING:
    from brew.envs.shell_env.base import ShellEnvironment as BaseEnvironment

logger = logging.getLogger(__name__)


class CodeActAgent(BaseAgent):
    """
    A CodeAct agent that solves coding tasks using function calling.

    The agent maintains a conversation history and uses tools (bash, editor, finish)
    to interact with an environment.
    """

    def __init__(
        self,
        environment: "BaseEnvironment",
        config: Optional[AgentConfig] = None,
        tools: Optional[list[BaseTool]] = None,
    ):
        """
        Initialize the CodeAct agent.

        Args:
            environment: Environment for executing commands
            config: Agent configuration
            tools: Optional list of tools (defaults to bash, editor, finish)
        """
        super().__init__(environment, config)
        self._tools = tools or get_default_tools(
            workspace_mount_path_in_sandbox=getattr(environment, "workspace_dir", None)
        )
        self._tool_map = {tool.name: tool for tool in self._tools}

    @property
    def tools(self) -> list[BaseTool]:
        """Return the list of tools available to the agent."""
        return self._tools

    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return get_core_system_prompt(self.config.model)

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get the OpenAI-compatible tool schemas for all tools."""
        short_desc_model_substrs = ["gpt-4", "o3", "o1", "o4"]
        use_short_description = any(
            substr in self.config.model for substr in short_desc_model_substrs
        )
        schemas = [
            tool.to_openai_schema(use_short_description=use_short_description)
            for tool in self.tools
        ]
        return schemas

    def _process_tool_call(self, tool_call: Any) -> str:
        """Process a single tool call and return the result."""
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return f"Error parsing tool arguments: {e}"

        logger.info(f"Tool call: {name} with args: {arguments}")

        if name not in self._tool_map:
            return f"Error: Unknown tool '{name}'"

        tool = self._tool_map[name]
        try:
            result = tool.execute(self.environment, **arguments)
            return result.output
        except FinishSignal as e:
            self.finished = True
            self.finish_message = e.message
            return e.message

    def _step(self) -> bool:
        """
        Execute one step of the agent loop.

        Returns:
            True if the agent should continue, False if finished or error
        """
        try:
            response = self._call_llm()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return False

        choice = response.choices[0]
        assistant_message = choice.message

        # Add assistant message to history
        msg_dict: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message.content,
        }
        if hasattr(assistant_message, "reasoning_content"):
            msg_dict["reasoning_content"] = assistant_message.reasoning_content

        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ]
        self.messages.append(msg_dict)

        # Process tool calls if present
        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                result = self._process_tool_call(tool_call)

                # Add tool result to history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                logger.info(f"Tool result: {result}")

                if self.finished:
                    return False

        return True


def run_agent(
    task: str,
    image: str,
    config: Optional[AgentConfig] = None,
    workspace_dir: str = "/workspace",
    environment: str = "docker",
) -> "AgentResult":
    """
    Convenience function to run an agent on a task.

    Args:
        task: The task description
        image: Container image to use
        config: Optional agent configuration
        workspace_dir: Working directory in the container
        environment: Execution environment type ("docker" or "enroot")

    Returns:
        AgentResult with the outcome
    """
    if environment == "docker":
        from brew.envs.shell_env.docker import DockerEnvironment

        runner = DockerEnvironment(image=image, workspace_dir=workspace_dir)
    elif environment == "enroot":
        from brew.envs.shell_env.enroot import EnrootEnvironment

        runner = EnrootEnvironment(image=image, workspace_dir=workspace_dir)
    else:
        raise ValueError(f"Unsupported environment: {environment}")

    try:
        runner.start()
        agent = CodeActAgent(environment=runner, config=config)
        result = agent.run(task)
        return result
    finally:
        runner.stop()
