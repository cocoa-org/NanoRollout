"""OpenHands-specific tool abstractions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from nanorollout.envs.shell_env.types import ToolResult

if TYPE_CHECKING:
    from nanorollout.envs.shell_env.base import ShellEnvironment as BaseEnvironment


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: str
    required: bool = False
    enum: Optional[list[str]] = None
    items: Optional[dict[str, Any]] = None
    default: Optional[Any] = None


class FinishSignal(Exception):
    """Exception raised by a finish tool to signal task completion."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class BaseTool(ABC):
    """Abstract base class for OpenHands tools exposed to agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """List of parameters the tool accepts."""
        ...

    @abstractmethod
    def get_description(self, **kwargs) -> str:
        """Get a description of what the tool does."""
        ...

    def execute(self, environment: "BaseEnvironment", **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.

        Override this method for tools that don't need the environment.
        """
        return environment.execute_tool(self.name, **kwargs)

    def to_openai_schema(self, **kwargs) -> dict[str, Any]:
        """Generate an OpenAI-compatible function schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.get_description(**kwargs),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
