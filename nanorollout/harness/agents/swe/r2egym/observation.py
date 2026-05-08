"""
Observation formatting  (stolen from r2egym/agenthub/observation/observation.py)
"""

from .prompts import CONTINUE_MSG

_OBS_TRUNCATE_LINES = 40
_MAX_OBS_CHARS = 20000


def _truncate_chars(text: str, limit: int = _MAX_OBS_CHARS) -> str:
    """Hard character-level truncation as a safety net."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + "\n<response clipped due to length>\n"
        + text[-half:]
    )


def _format_observation(
    function_name: str,
    bash_output: str,
    error_code: int,
) -> str:
    """Format the environment output the way R2E-Gym presents it to the LLM."""
    if not function_name:
        return CONTINUE_MSG
    if function_name in ("finish", "submit"):
        return "<<< Finished >>>"
    if function_name in ("execute_bash", "bash"):
        lines = bash_output.splitlines() if bash_output else []
        if len(lines) > 2 * _OBS_TRUNCATE_LINES:
            top = "\n".join(lines[:_OBS_TRUNCATE_LINES])
            bottom = "\n".join(lines[-_OBS_TRUNCATE_LINES:])
            divider = "-" * 50
            truncated = (
                f"{top}\n{divider}\n"
                "<Observation truncated in middle for saving context>\n"
                f"{divider}\n{bottom}"
            )
        else:
            truncated = bash_output
        result = (
            f"Exit code: {error_code}\n"
            f"Execution output of [{function_name}]:\n{truncated}"
        )
        return _truncate_chars(result)
    return _truncate_chars(f"Execution output of [{function_name}]:\n{bash_output}")
