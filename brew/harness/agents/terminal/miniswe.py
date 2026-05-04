"""MiniSweAgent subclass for Terminal Bench tasks."""

from typing import Any

from jinja2 import BaseLoader, Environment as JinjaEnvironment

from harness.agents.prompts.terminal import (
    INSTANCE_TEMPLATE,
    SYSTEM_TEMPLATE,
)
from harness.agents.swe.miniswe import MiniSweAgent

_jinja_env = JinjaEnvironment(loader=BaseLoader())


def _render(template_str: str, **kwargs: Any) -> str:
    tmpl = _jinja_env.from_string(template_str)
    return tmpl.render(**kwargs)


class TerminalMiniSweAgent(MiniSweAgent):
    """MiniSweAgent with Terminal Bench prompts instead of SWE-Bench prompts."""

    def get_system_prompt(self) -> str:
        return SYSTEM_TEMPLATE

    def _init_messages(self, task: str) -> None:
        user_content = _render(INSTANCE_TEMPLATE, task=task)
        self.messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_content},
        ]
