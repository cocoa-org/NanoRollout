"""OSWorld adapter package."""

from .adapter import OSWorldTaskAdapter
from .entrypoints import run_osworld

__all__ = ["OSWorldTaskAdapter", "run_osworld"]
