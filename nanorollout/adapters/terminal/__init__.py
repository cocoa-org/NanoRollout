"""Terminal task adapter package."""

from .entrypoints import (
    run_tb_claude_code,
    run_tb_miniswe,
    run_tb_opencode,
    run_tb_qwen_code,
    run_tb_terminus2,
)

__all__ = [
    "run_tb_claude_code",
    "run_tb_miniswe",
    "run_tb_opencode",
    "run_tb_qwen_code",
    "run_tb_terminus2",
]
