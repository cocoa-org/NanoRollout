"""
Utility functions for the executor environment.
"""

import json
import os
import time
import requests
from typing import Any, Dict
from colorama import Fore, Style

from .logger import setup_logging, get_logger

__all__ = ["setup_logging", "get_logger", "colorize"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    logger = get_logger("config")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}


def retry_request(func, max_retries: int = 3, delay: float = 1.0):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))


def validate_response(response: requests.Response) -> Dict[str, Any]:
    """Validate and parse HTTP response."""
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP error: {e}")
    except json.JSONDecodeError:
        raise Exception("Invalid JSON response")


def measure_execution_time(func):
    """Decorator to measure function execution time and add to result if it's a dict."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Add execution_time to dict results
        if isinstance(result, dict):
            result['execution_time'] = execution_time

        return result
    return wrapper


def extract_config_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant config information for result recording."""
    controller_config = config.get("controller", {})
    sandbox_config = config.get("sandbox", {})

    config_info = {
        "controller": controller_config,
        "sandbox": sandbox_config
    }

    config_info['controller']['args'].pop("api_key", None)

    return config_info


def colorize(obj: Any, color: str = "CYAN") -> str:
    """Apply color to an object for terminal output.

    Args:
        obj: The object to colorize (will be converted to string)
        color: The color name (e.g., "CYAN", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "WHITE")
               Defaults to "CYAN"

    Returns:
        The string representation of the object wrapped with color codes

    Example:
        >>> colorized = colorize("Hello", "GREEN")
        >>> print(colorized)  # Will print in green
    """
    color_upper = color.upper()

    # Get the color from colorama.Fore
    color_code = getattr(Fore, color_upper, Fore.CYAN)

    # Convert object to string and wrap with color codes
    return f"{color_code}{str(obj)}{Style.RESET_ALL}"
