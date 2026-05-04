import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm import completion, token_counter
from litellm import completion_cost

if TYPE_CHECKING:
    from brew.envs.shell_env.base import ShellEnvironment as BaseEnvironment
    from brew.harness.tools.base import BaseTool

logger = logging.getLogger(__name__)

EXIT_STATUS_FINISHED = "finished"  # agent output finished
EXIT_STATUS_MAX_LENGTH = "max_length"
EXIT_STATUS_MAX_ITERATIONS = "max_iterations"
EXIT_STATUS_ERROR = "error"
EXIT_STATUS_UNKNOWN = "unknown"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    model: str = "claude-sonnet-4-20250514"
    max_iterations: int = 30
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048
    extra_body: dict[str, Any] = field(default_factory=dict)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    llm_provider: str = "openai"


@dataclass
class AgentResult:
    """Result of an agent run."""

    success: bool
    message: str
    history: list[dict[str, Any]] = field(default_factory=list)
    llm_metrics: list[dict[str, Any]] = field(default_factory=list)
    llm_cost_total: float = 0.0
    patch: str = ""
    iterations: int = 0
    error: Optional[str] = None
    exit_status: str = EXIT_STATUS_UNKNOWN


class BaseAgent(ABC):
    """
    Abstract base class for agents.

    An agent uses tools to interact with an environment to accomplish tasks.
    Subclasses must implement the tools property, get_system_prompt method,
    and _step method.
    """

    environment: "BaseEnvironment"
    config: AgentConfig
    messages: list[dict[str, Any]]
    finished: bool
    finish_message: str

    def __init__(
        self,
        environment: "BaseEnvironment",
        config: Optional[AgentConfig] = None,
    ):
        self.environment = environment
        self.config = config or AgentConfig()
        self.messages: list[dict[str, Any]] = []
        self.llm_metrics: list[dict[str, Any]] = []
        self.finished = False
        self.finish_message = ""
        self.exit_status = EXIT_STATUS_UNKNOWN

    @property
    @abstractmethod
    def tools(self) -> list["BaseTool"]:
        """Return the list of tools available to the agent."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        ...

    @abstractmethod
    def _step(self) -> bool:
        """Execute one step of the agent loop. Returns True to continue."""
        ...

    def _init_messages(self, task: str) -> None:
        """Initialize the conversation with system prompt and task."""
        self.messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": task},
        ]

    def _get_continuation_prompt(self) -> str:
        """Get the prompt to continue the agent when it stops without finishing."""
        return (
            "Please continue working on the task. "
            "If you think you have solved the problem, use the finish tool. "
            "IMPORTANT: You should NEVER ask for human help."
        )

    def _needs_continuation(self) -> bool:
        """Check if the agent needs a continuation prompt."""
        if not self.messages:
            return False
        last_msg = self.messages[-1]
        if last_msg["role"] == "assistant":
            if "tool_calls" not in last_msg or not last_msg["tool_calls"]:
                return True
        return False

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get the OpenAI-compatible tool schemas for all tools."""
        return [tool.to_openai_schema() for tool in self.tools]
    
    def _call_llm(self) -> dict[str, Any]:
        """Make a completion call to the LLM."""
        kwargs = {
            "model": self.config.model,
            "messages": self.messages,
            "tools": self.get_tools_schema(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "extra_body": self.config.extra_body,
        }

        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.llm_provider:
            kwargs["custom_llm_provider"] = self.config.llm_provider
        elif self.config.api_base and "/" not in self.config.model:
            kwargs["custom_llm_provider"] = "openai"

        effort = getattr(self.config, "reasoning_effort", None)
        if effort:
            kwargs["reasoning_effort"] = effort
            kwargs["extra_body"]["reasoning_effort"] = effort

        try:
            response = completion(**kwargs, drop_params=True)
        except litellm.ContextWindowExceededError as e:
            # The input is longer than the model's context length
            logger.error(f"Context window exceeded: {e}")
            self.exit_status = EXIT_STATUS_MAX_LENGTH
            raise
        except litellm.BadRequestError as e:
            # Requested token count exceeds the model's maximum context length
            logger.error(f"Request token limit exceeded: {e}")
            self.exit_status = EXIT_STATUS_MAX_LENGTH
            raise

        self._record_llm_metrics(response)
        try:
            content = response.choices[0].message.content
            logger.info(f"LLM response: {response.choices[0].message.content}")
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return None
        return response
    
    def _record_llm_metrics(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        usage_dict = None
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif hasattr(usage, "dict"):
                usage_dict = usage.dict()
            elif isinstance(usage, dict):
                usage_dict = dict(usage)
            else:
                usage_dict = {"raw": str(usage)}

        cost = getattr(response, "cost", None)
        hidden = getattr(response, "_hidden_params", None)
        if cost is None and isinstance(hidden, dict):
            for key in ("cost", "response_cost", "total_cost"):
                if key in hidden:
                    cost = hidden[key]
                    break

        if cost is None and completion_cost is not None:
            try:
                cost = completion_cost(response=response)
            except Exception:
                cost = None

        self.llm_metrics.append(
            {
                "model": getattr(response, "model", None),
                "usage": usage_dict,
                "cost": cost,
            }
        )

    def run(self, task: str) -> AgentResult:
        """Run the agent on a task."""
        self._init_messages(task)
        self.finished = False
        self.finish_message = ""
        self.llm_metrics = []

        iterations = 0
        error = None

        try:
            while iterations < self.config.max_iterations:
                iterations += 1
                logger.info(f"Iteration {iterations}/{self.config.max_iterations}")

                should_continue = self._step()
                if not should_continue:
                    break

                if self._needs_continuation():
                    self.messages.append({
                        "role": "user",
                        "content": self._get_continuation_prompt(),
                    })
            
            if self.finished:
                self.exit_status = EXIT_STATUS_FINISHED
            elif iterations == self.config.max_iterations:
                self.exit_status = EXIT_STATUS_MAX_ITERATIONS

        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            error = str(e)
            self.exit_status = EXIT_STATUS_ERROR

        patch = self.environment.get_git_diff()
        llm_cost_total = sum(
            metric.get("cost", 0.0)
            for metric in self.llm_metrics
            if isinstance(metric.get("cost"), (int, float))
        )

        return AgentResult(
            success=self.finished,
            message=self.finish_message,
            history=self.messages,
            llm_metrics=self.llm_metrics,
            llm_cost_total=llm_cost_total,
            patch=patch,
            iterations=iterations,
            error=error,
            exit_status=self.exit_status,
        )
