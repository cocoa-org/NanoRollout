"""Cocoa agent integrations for NanoRollout."""

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
    if name == "CocoaAgent":
        from .cocoa_agent import CocoaAgent

        return CocoaAgent
    raise AttributeError(name)

__all__ = [
    "BaseAgent",
    "BaseLLM",
    "ClaudeLLM",
    "Controller",
    "CocoaAgent",
    "DeepSeekLLM",
    "GeminiLLM",
    "GLMLLM",
    "Human",
    "KimiLLM",
    "OpenAILLM",
    "QwenLLM",
]
