"""Cocoa-Bench runner entry points."""

from .cocoa_agent import run_cocoa_agent
from .installed import (
    run_cocoa_claude_code,
    run_cocoa_opencode,
    run_cocoa_qwen_code,
)

__all__ = [
    "run_cocoa_agent",
    "run_cocoa_claude_code",
    "run_cocoa_qwen_code",
    "run_cocoa_opencode",
]
