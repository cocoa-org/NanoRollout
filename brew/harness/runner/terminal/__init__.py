"""Terminal benchmark runner entry points."""

from .miniswe import run_tb_miniswe
from .terminus2 import run_tb_terminus2

__all__ = ["run_tb_miniswe", "run_tb_terminus2"]
