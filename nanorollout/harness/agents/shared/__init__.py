"""Shared agent integrations reused across NanoRollout harnesses."""

from .claude_code import ClaudeCode
from .installed_agent_factory import (
    AGENT_REGISTRY,
    build_installed_agent,
    inject_agent_credentials,
)
from .llm_config import LLMConfig, build_llm_config, parse_json_object
from .opencode import OpenCode
from .qwen_code import QwenCode

__all__ = [
    "AGENT_REGISTRY",
    "ClaudeCode",
    "LLMConfig",
    "OpenCode",
    "QwenCode",
    "build_installed_agent",
    "build_llm_config",
    "inject_agent_credentials",
    "parse_json_object",
]
