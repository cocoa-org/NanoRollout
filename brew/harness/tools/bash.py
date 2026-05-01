"""
Bash tool implementation.
"""

from .base import BaseTool, ToolParameter

_DETAILED_BASH_DESCRIPTION = """Execute a bash command in the terminal within a persistent shell session.


### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Default timeout: Commands use a hard timeout (defaults to the environment's timeout if you don't set `timeout`).
* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. The runtime may not support them and can cause unusable shell sessions. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.

### Long-running Commands
* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
* For commands that may run for a long time (e.g. installation or testing commands), or commands that run for a fixed amount of time (e.g. sleep), you should set the "timeout" parameter of your function call to an appropriate value.
* If a bash command returns exit code `-1`, this means the process hit the timeout and is not yet finished. By setting `is_input` to `true`, you can:
  - Send empty `command` to retrieve additional logs
  - Send text (set `command` to the text) to STDIN of the running process
  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process
  - If you do C-c, you can re-start the process with a longer "timeout" parameter to let it run to completion

### Best Practices
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.
"""

_SHORT_BASH_DESCRIPTION = """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`. For commands that need to run for a specific duration, you can set the "timeout" argument to specify a hard timeout in seconds (defaults to the environment's timeout).
* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, the assistant can interact with the running process and send empty `command` to retrieve any additional logs, or send additional text (set `command` to the text) to STDIN of the running process, or send command like `C-c` (Ctrl+C), `C-d` (Ctrl+D), `C-z` (Ctrl+Z) to interrupt the process.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."""


class BashTool(BaseTool):
    """Tool for executing bash commands in the environment."""

    def execute(self, environment, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = getattr(environment, "timeout", None)
        return super().execute(environment, **kwargs)

    @property
    def name(self) -> str:
        return "execute_bash"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.",
                required=True,
            ),
            ToolParameter(
                name="is_input",
                type="string",
                description="If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.",
                required=False,
                enum=["true", "false"],
            ),
            ToolParameter(
                name="timeout",
                type="number",
                description="Optional. Sets a hard timeout in seconds for the command execution. If not provided, defaults to the environment's timeout.",
                required=False,
            ),
        ]

    def get_description(self, **kwargs) -> str:
        return (
            _SHORT_BASH_DESCRIPTION
            if kwargs.get("use_short_description", False)
            else _DETAILED_BASH_DESCRIPTION
        )
