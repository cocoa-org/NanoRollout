"""
Unified logging configuration for the executor environment.
Provides consistent log formatting and logger management across all modules.
"""

import logging
from typing import Optional
from colorama import Fore, Style

DATE_FORMAT = "%H:%M:%S"

LEVEL_COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}

_setup_done = False
_root_logger: Optional[logging.Logger] = None


class ColoredFormatter(logging.Formatter):
    """Formatter that colorizes the log prefix (timestamp, name, level)."""

    def format(self, record):
        # Format timestamp
        timestamp = self.formatTime(record, self.datefmt)

        # Build colored prefix: timestamp - name:level:
        prefix = f"{timestamp} - {record.name}:{record.levelname}:"
        color = LEVEL_COLORS.get(record.levelname, Fore.WHITE)
        colored_prefix = f"{color}{prefix}{Style.RESET_ALL}"

        # Return colored prefix + filename:lineno - message
        return f"{colored_prefix} {record.filename}:{record.lineno} - {record.getMessage()}"


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the executor application."""
    global _setup_done, _root_logger

    level = getattr(logging, log_level.upper(), logging.INFO)

    _root_logger = logging.getLogger("executor")
    _root_logger.setLevel(level)
    _root_logger.handlers.clear()
    _root_logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(datefmt=DATE_FORMAT))
    _root_logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)d - %(message)s",
                datefmt=DATE_FORMAT,
            )
        )
        _root_logger.addHandler(file_handler)

    _setup_done = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with namespace 'executor' or 'executor.{name}'."""
    if not _setup_done:
        setup_logging()
    return logging.getLogger(f"executor.{name}" if name else "executor")
