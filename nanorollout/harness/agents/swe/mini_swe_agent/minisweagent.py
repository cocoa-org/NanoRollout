"""
MiniSweAgent implementation.

A minimalist SWE agent that uses text-based prompting with bash code blocks
instead of LLM function calling. Based on the mini-swe-agent approach that
achieves 74%+ on SWE-bench verified using only bash commands.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from jinja2 import BaseLoader, Environment as JinjaEnvironment

from .prompts import (
    ACTION_OBSERVATION_TEMPLATE,
    ACTION_PATTERN,
    FINISH_SIGNALS,
    FORMAT_ERROR_TEMPLATE,
    INSTANCE_TEMPLATE,
    SYSTEM_TEMPLATE,
    TIMEOUT_TEMPLATE,
)
from ..base import AgentConfig, BaseAgent

if TYPE_CHECKING:
    from nanorollout.envs.shell_env.base import ShellEnvironment as BaseEnvironment

logger = logging.getLogger(__name__)

_jinja_env = JinjaEnvironment(loader=BaseLoader())


def _render(template_str: str, **kwargs: Any) -> str:
    """Render a Jinja2 template string with the given variables."""
    tmpl = _jinja_env.from_string(template_str)
    return tmpl.render(**kwargs)


class MiniSweAgent(BaseAgent):
    """
    A minimalist SWE agent that uses text-based bash prompting.

    Instead of LLM function calling, the agent instructs the model to output
    a THOUGHT section followed by a single ```bash``` code block. The agent
    parses the bash command via regex and executes it directly in the
    environment.
    """

    def __init__(
        self,
        environment: "BaseEnvironment",
        config: Optional[AgentConfig] = None,
        step_timeout: int = 60,
    ):
        super().__init__(environment, config)
        self.step_timeout = step_timeout

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[Any]:
        """No tools — this agent uses text-based bash commands."""
        return []

    def get_system_prompt(self) -> str:
        return SYSTEM_TEMPLATE

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """No tool schemas — text-based agent."""
        return []

    # ------------------------------------------------------------------
    # Message initialisation
    # ------------------------------------------------------------------

    def _init_messages(self, task: str) -> None:
        """Build initial messages using the swebench templates."""
        user_content = _render(INSTANCE_TEMPLATE, task=task)
        self.messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Continuation — not needed for this agent
    # ------------------------------------------------------------------

    def _needs_continuation(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _step(self) -> bool:
        """
        Execute one agent step:
        1. Call LLM (text-based, no tools)
        2. Parse bash action from response
        3. Execute action in environment
        4. Check for finish signal
        5. Format observation and append to messages
        """
        # 1. Call LLM
        try:
            response = self._call_llm()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return False

        msg = response.choices[0].message
        assistant_content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", None) or ""

        # Prepend reasoning as <think> block so it stays in conversation history
        if reasoning_content:
            full_content = f"<think>\n{reasoning_content}\n</think>\n\n{assistant_content}"
        else:
            full_content = assistant_content

        logger.info(f"LLM response: {full_content}")

        # 2. Append assistant message (with thinking included)
        self.messages.append({"role": "assistant", "content": full_content})

        # 3. Parse bash action(s) via regex
        actions = ACTION_PATTERN.findall(assistant_content)

        if len(actions) != 1:
            # Format error: 0 or >1 bash blocks
            error_msg = _render(
                FORMAT_ERROR_TEMPLATE, actions_count=len(actions)
            )
            self.messages.append({"role": "user", "content": error_msg})
            return True  # continue — let the LLM retry

        action = actions[0].strip()

        # 4. Execute bash action
        try:
            result = self.environment.execute(action, timeout=self.step_timeout)
            output = result.output
            returncode = result.exit_code
            timed_out = False
        except Exception as e:
            error_str = str(e)
            if "timeout" in error_str.lower() or "timed out" in error_str.lower():
                timed_out = True
                output = error_str
                returncode = -1
            else:
                raise

        # Handle timeout
        if timed_out:
            timeout_msg = _render(
                TIMEOUT_TEMPLATE, action=action, output=output
            )
            self.messages.append({"role": "user", "content": timeout_msg})
            return True  # continue

        # 5. Check finish signal
        for signal in FINISH_SIGNALS:
            if signal in output:
                self.finished = True
                self.finish_message = output
                return False

        # 6. Format observation and append
        observation = _render(
            ACTION_OBSERVATION_TEMPLATE,
            returncode=returncode,
            output=output,
        )
        self.messages.append({"role": "user", "content": observation})

        return True  # continue
