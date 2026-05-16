"""SWE task adapter package."""

from .entrypoints import (
    run_installed_claude_code,
    run_installed_opencode,
    run_installed_qwen_code,
    run_miniswe,
    run_oh_core,
    run_oh_lite,
    run_r2egym,
)

__all__ = [
    "run_installed_claude_code",
    "run_installed_opencode",
    "run_installed_qwen_code",
    "run_miniswe",
    "run_oh_core",
    "run_oh_lite",
    "run_r2egym",
]
