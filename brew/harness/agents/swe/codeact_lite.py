"""
CodeAct-Lite Agent — minimal-prompt variant of CodeActAgent.

Uses the lightweight ProRL-style system prompt (oh_lite) to save token budget
for actual code interactions, which is better suited for smaller models.
"""

from typing import TYPE_CHECKING, Optional

from brew.harness.agents.prompts.swe.oh_lite import get_system_prompt
from brew.harness.agents.swe.codeact import CodeActAgent
from brew.harness.agents.swe.base import AgentConfig
from brew.harness.tools.base import BaseTool

if TYPE_CHECKING:
    from brew.envs.base import BaseEnvironment


class CodeActLiteAgent(CodeActAgent):
    """CodeActAgent with a minimal system prompt."""

    def get_system_prompt(self) -> str:
        return get_system_prompt(self.config.model)
