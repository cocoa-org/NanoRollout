"""UDA agent integrations for NanoRollout.

Mirrors :mod:`nanorollout.harness.agents.cocoa` but binds to
:mod:`nanorollout.envs.uda_env` instead of ``cocoa_env``. The LLM
controllers (OpenAILLM / ClaudeLLM / QwenLLM / ...) are reused verbatim
because the per-provider request / response handling is independent of
which sandbox surface the tools target; the only UDA-specific changes
live in the controller's system prompt (tool vocabulary) and in the
tool schema imports (``uda_env.tools`` exposes ``computer_use_*``
instead of ``browser_*`` / ``dom_*``).
"""

from .base import BaseAgent
from .controller import (
    BaseLLM,
    ClaudeLLM,
    Controller,
    DeepSeekLLM,
    GeminiLLM,
    GLMLLM,
    Human,
    KimiLLM,
    OpenAILLM,
    QwenLLM,
)


def __getattr__(name: str):
    if name == "UDAAgent":
        from .uda_agent import UDAAgent

        return UDAAgent
    raise AttributeError(name)


__all__ = [
    "BaseAgent",
    "BaseLLM",
    "ClaudeLLM",
    "Controller",
    "UDAAgent",
    "DeepSeekLLM",
    "GeminiLLM",
    "GLMLLM",
    "Human",
    "KimiLLM",
    "OpenAILLM",
    "QwenLLM",
]
