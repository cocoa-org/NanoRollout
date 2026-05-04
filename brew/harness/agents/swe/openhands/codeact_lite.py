"""
CodeAct-Lite Agent implementation.
"""

from .codeact import CodeActAgent
from .prompts import get_lite_system_prompt


class CodeActLiteAgent(CodeActAgent):
    """CodeActAgent with a minimal system prompt."""

    def get_system_prompt(self) -> str:
        return get_lite_system_prompt(self.config.model)
