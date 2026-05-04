"""
Shell/filesystem environment primitives.
"""

import logging
import os
import re
import shlex
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from brew.harness.tools.base import ToolResult

GIT_USER_EMAIL = "evaluation@openhands.dev"
GIT_USER_NAME = "OpenHands Evaluation"
GIT_COMMIT_MESSAGE = "patch"


@dataclass
class ExecutionResult:
    """Result of a command execution."""

    output: str
    exit_code: int


DEFAULT_TRUNCATE_NOTICE = "<response clipped>"
DEFAULT_TRUNCATE_NOTICE_WITH_PERSIST = (
    "<response clipped; full output saved to {file_path} at line {line_num}>"
)
TOOL_LOGGER_NAME = "brew.envs.tools"


def extract_cwd_marker(output: str, marker: str) -> tuple[str, Optional[str]]:
    """Extract a trailing cwd marker from command output.

    Only treat the marker as valid when the value after it looks like an
    absolute path. This avoids corrupting ``_cwd`` when error messages happen
    to include the marker string, such as timeout exceptions that echo the
    wrapped shell command.
    """
    if marker not in output:
        return output, None

    before, after = output.rsplit(marker, 1)
    candidate = after.strip().splitlines()[0] if after.strip() else ""
    if not os.path.isabs(candidate):
        return output, None

    return before.rstrip("\n"), candidate


def _save_full_content(content: str, save_dir: str, tool_prefix: str) -> Optional[str]:
    try:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{tool_prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.txt"
        path = os.path.join(save_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path
    except Exception:
        return None


def maybe_truncate(
    content: str,
    truncate_after: Optional[int] = None,
    truncate_notice: str = DEFAULT_TRUNCATE_NOTICE,
    save_dir: Optional[str] = None,
    tool_prefix: str = "output",
) -> str:
    if not truncate_after or len(content) <= truncate_after or truncate_after < 0:
        return content

    if len(truncate_notice) >= truncate_after:
        return truncate_notice[:truncate_after]

    available_chars = truncate_after - len(truncate_notice)
    proposed_head = available_chars // 2 + (available_chars % 2)

    final_notice = truncate_notice
    if save_dir:
        saved_file_path = _save_full_content(content, save_dir, tool_prefix)
        if saved_file_path:
            head_content_lines = len(content[:proposed_head].splitlines())
            final_notice = DEFAULT_TRUNCATE_NOTICE_WITH_PERSIST.format(
                file_path=saved_file_path,
                line_num=head_content_lines + 1,
            )

    if len(final_notice) >= truncate_after:
        return final_notice[:truncate_after]

    remaining = truncate_after - len(final_notice)
    head_chars = min(proposed_head, remaining)
    tail_chars = remaining - head_chars

    return (
        content[:head_chars]
        + final_notice
        + (content[-tail_chars:] if tail_chars > 0 else "")
    )


class ShellEnvironment(ABC):
    """
    Abstract base class for shell-backed execution environments.

    A shell environment provides command execution, workspace filesystem
    operations, and shell-oriented tool adapters. Implementations include
    Docker, Enroot, Modal, or other command-capable sandboxes.
    """

    workspace_dir: str
    timeout: int
    max_output_chars: int = 30000
    MAX_FILE_SIZE_MB: int = 10
    MAX_HISTORY_PER_FILE: int = 10
    SNIPPET_CONTEXT_WINDOW: int = 3

    @abstractmethod
    def start(self) -> None:
        """Start the environment."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop and clean up the environment."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the environment is currently running."""
        ...

    @abstractmethod
    def execute(self, command: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Execute a command in the environment."""
        ...

    def set_tool_log_context(self, context: Optional[str]) -> None:
        """Set a context prefix for tool execution logs."""
        self._tool_log_context = context or ""

    def _tool_log_prefix(self) -> str:
        context = getattr(self, "_tool_log_context", "")
        return f"[{context}] " if context else ""

    def execute_tool(self, tool_name: str, **kwargs) -> "ToolResult":
        """
        Execute a tool by name with given arguments.

        Dispatches to _tool_{tool_name} method if it exists.
        """

        handler_name = f"_tool_{tool_name}"
        handler = getattr(self, handler_name, None)
        tool_logger = logging.getLogger(TOOL_LOGGER_NAME)
        log_prefix = self._tool_log_prefix()

        if handler is None:
            if tool_logger.isEnabledFor(logging.WARNING):
                tool_logger.warning(
                    "%stool_missing name=%s", log_prefix, tool_name
                )
            return ToolResult(
                output=f"Error: Tool '{tool_name}' is not supported by this environment",
                success=False,
            )

        if tool_logger.isEnabledFor(logging.INFO):
            tool_logger.info(
                "%stool_start name=%s args=%s",
                log_prefix,
                tool_name,
                kwargs,
            )
        started = time.time()
        try:
            result = handler(**kwargs)
        except Exception:
            if tool_logger.isEnabledFor(logging.ERROR):
                tool_logger.exception("%stool_error name=%s", log_prefix, tool_name)
            raise
        duration = time.time() - started
        if tool_logger.isEnabledFor(logging.INFO):
            metadata = result.metadata
            log_output = metadata.get("log_output", result.output)
            metadata_for_log = (
                {key: value for key, value in metadata.items() if key != "log_output"}
                if metadata
                else metadata
            )
            tool_logger.info(
                "%stool_end name=%s success=%s duration=%.2fs metadata=%s output_chars=%s output=%s",
                log_prefix,
                tool_name,
                result.success,
                duration,
                metadata_for_log,
                len(log_output),
                log_output,
            )
        return result

    def _tool_execute_bash(
        self,
        command: str,
        is_input: str = "false",
        timeout: Optional[int] = None,
        security_risk: Optional[str] = None,
        **kwargs,
    ) -> "ToolResult":
        """Execute bash command."""

        if timeout is not None and timeout > 120:
            timeout = 120

        if is_input == "true" and not command:
            return ToolResult(
                output="ERROR: No previous running command to retrieve logs from.",
                success=False,
            )
        if is_input == "true" and command:
            return ToolResult(
                output="ERROR: Interactive input is not supported by this environment.",
                success=False,
            )

        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        if "pytest" in tokens and "-q" in tokens:
            return ToolResult(
                output=(
                    "ERROR: Do not run `pytest -q` in this environment because it is "
                    "extremely slow. Please run pytest without `-q` or use a more "
                    "targeted command."
                ),
                success=False,
            )

        _BLOCKED_CMDS = {"kill", "pkill", "killall", "skill", "xkill"}
        if _BLOCKED_CMDS & set(tokens):
            return ToolResult(
                output="ERROR: Do not try to kill processes in this environment as it will affect other tasks.",
                success=False,
            )

        result = self.execute(command, timeout=timeout)
        raw_output = result.output.rstrip("\n")

        output = self._truncate_output(result.output)
        output = output.rstrip("\n")

        suffix_lines: list[str] = []
        cwd = getattr(self, "_cwd", None)
        if cwd:
            suffix_lines.append(f"[Current working directory: {cwd}]")
        if result.exit_code != -1:
            suffix_lines.append(
                f"[Command finished with exit code {result.exit_code}]"
            )

        if suffix_lines:
            output = f"{output}\n" if output else ""
            output += "\n".join(suffix_lines)

        return ToolResult(
            output=output,
            success=result.exit_code == 0,
            metadata={"exit_code": result.exit_code, "is_input": is_input},
        )

    def _tool_str_replace_editor(
        self,
        command: str,
        path: str,
        file_text: Optional[str] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        view_range: Optional[list[int]] = None,
        security_risk: Optional[str] = None,
        **kwargs,
    ) -> "ToolResult":
        """Execute editor command."""

        path_error = self._editor_validate_path(command, path)
        if path_error is not None:
            return path_error

        if command == "view":
            return self._editor_view(path, view_range)
        if command == "create":
            return self._editor_create(path, file_text)
        if command == "str_replace":
            return self._editor_str_replace(path, old_str, new_str)
        if command == "insert":
            return self._editor_insert(path, insert_line, new_str)
        if command == "undo_edit":
            return self._editor_undo(path)

        return ToolResult(
            output=f"Error: Unknown editor command '{command}'",
            success=False,
        )

    def _editor_path_exists(self, path: str) -> bool:
        result = self.execute(f"test -e {shlex.quote(path)}")
        return result.exit_code == 0

    def _editor_validate_path(self, command: str, path: str) -> Optional["ToolResult"]:

        if not os.path.isabs(path):
            suggestion_message = "The path should be an absolute path, starting with `/`."
            cwd = getattr(self, "_cwd", None) or getattr(self, "workspace_dir", None)
            if cwd:
                suggested_path = os.path.join(cwd, path)
                if self._editor_path_exists(suggested_path):
                    suggestion_message += f" Maybe you meant {suggested_path}?"
            return ToolResult(output=suggestion_message, success=False)

        if command == "create":
            if self._editor_path_exists(path):
                return ToolResult(
                    output=(
                        f"File already exists at: {path}. Cannot overwrite files using "
                        "command `create`."
                    ),
                    success=False,
                )
            return None

        if not self._editor_path_exists(path):
            return ToolResult(
                output=f"The path {path} does not exist. Please provide a valid path.",
                success=False,
            )
        if command != "view" and self.is_directory(path):
            return ToolResult(
                output=(
                    f"The path {path} is a directory and only the `view` command can "
                    "be used on directories."
                ),
                success=False,
            )
        return None

    def _editor_validate_file(self, path: str) -> Optional["ToolResult"]:

        if not self._editor_path_exists(path) or self.is_directory(path):
            return None

        file_size = self._editor_get_file_size(path)
        max_size = self.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size is not None and file_size > max_size:
            return ToolResult(
                output=(
                    f"File is too large ({file_size / 1024 / 1024:.1f}MB). "
                    f"Maximum allowed size is {int(max_size / 1024 / 1024)}MB."
                ),
                success=False,
            )

        return None

    def _editor_get_file_size(self, path: str) -> Optional[int]:
        path_quoted = shlex.quote(path)
        for cmd in (
            f"stat -c%s {path_quoted}",
            f"stat -f%z {path_quoted}",
            f"wc -c < {path_quoted}",
        ):
            result = self.execute(cmd)
            if result.exit_code != 0:
                continue
            value = result.output.strip().split()[0] if result.output else ""
            if value.isdigit():
                return int(value)
        return None

    def _editor_execute_python(self, code: str) -> Optional[ExecutionResult]:
        last_result: Optional[ExecutionResult] = None
        for python_cmd in ("python3", "python"):
            result = self.execute(f"{python_cmd} - <<'PY'\n{code}\nPY")
            last_result = result
            if result.exit_code == 0:
                return result
        return last_result

    def _editor_add_history(self, path: str, content: str) -> None:
        history = getattr(self, "_editor_history", None)
        if history is None:
            history = {}
            setattr(self, "_editor_history", history)
        entries = history.setdefault(path, [])
        entries.append(content)
        if len(entries) > self.MAX_HISTORY_PER_FILE:
            del entries[0]

    def _editor_pop_history(self, path: str) -> Optional[str]:
        history = getattr(self, "_editor_history", None)
        if not history:
            return None
        entries = history.get(path)
        if not entries:
            return None
        return entries.pop()

    def _editor_count_lines(self, content: str) -> int:
        return len(content.splitlines())

    def _editor_slice_content(self, content: str, start_line: int, end_line: int) -> str:
        lines = content.splitlines(keepends=True)
        return "".join(lines[start_line - 1 : end_line])

    def _editor_make_output(
        self, snippet_content: str, snippet_description: str, start_line: int = 1
    ) -> str:
        snippet_content = self._truncate_output(snippet_content)
        snippet_content = "\n".join(
            [
                f"{i + start_line:6}\t{line}"
                for i, line in enumerate(snippet_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {snippet_description}:\n"
            + snippet_content
            + "\n"
        )

    def _editor_view(
        self, path: str, view_range: Optional[list[int]] = None
    ) -> "ToolResult":
        """View a file or directory."""

        if self.is_directory(path):
            if view_range:
                return ToolResult(
                    output=(
                        "The `view_range` parameter is not allowed when `path` points "
                        "to a directory."
                    ),
                    success=False,
                )

            path_quoted = shlex.quote(path)
            hidden_result = self.execute(
                f"find -L {path_quoted} -mindepth 1 -maxdepth 1 -name '.*'"
            )
            hidden_count = (
                len(hidden_result.output.strip().splitlines())
                if hidden_result.output.strip()
                else 0
            )

            hidden1 = shlex.quote(f"{path}/.*")
            hidden2 = shlex.quote(f"{path}/*/.*")
            list_cmd = (
                f"find -L {path_quoted} -maxdepth 2 -not \\( -path {hidden1} -o "
                f"-path {hidden2} \\) | sort"
            )
            list_result = self.execute(list_cmd)
            if list_result.exit_code != 0:
                return ToolResult(output=list_result.output, success=False)

            paths = (
                list_result.output.strip().splitlines()
                if list_result.output.strip()
                else []
            )
            formatted_paths = []
            for entry in paths:
                if self.is_directory(entry):
                    formatted_paths.append(f"{entry}/")
                else:
                    formatted_paths.append(entry)
            msg = [
                f"Here's the files and directories up to 2 levels deep in {path}, "
                "excluding hidden items:\n" + "\n".join(formatted_paths)
            ]
            if hidden_count > 0:
                msg.append(
                    f"\n{hidden_count} hidden files/directories in this directory "
                    f"are excluded. You can use 'ls -la {path}' to see them."
                )
            output = "\n".join(msg)
            output = self._truncate_output(output)
            return ToolResult(output=output, success=True)

        # TODO: Add image support

        file_error = self._editor_validate_file(path)
        if file_error is not None:
            return file_error

        result = self.read_file(path)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to read {path}",
                success=False,
            )
        file_content = result.output
        num_lines = self._editor_count_lines(file_content)

        if not view_range:
            output = self._editor_make_output(file_content, str(path), 1)
            return ToolResult(output=output, success=True)

        if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
            return ToolResult(
                output="It should be a list of two integers.",
                success=False,
            )

        start_line, end_line = view_range
        if start_line < 1 or start_line > num_lines:
            return ToolResult(
                output=(
                    f"Its first element `{start_line}` should be within the range "
                    f"of lines of the file: {[1, num_lines]}."
                ),
                success=False,
            )

        # Normalize end_line and provide a warning if it exceeds file length
        warning_message: Optional[str] = None
        if end_line == -1:
            end_line = num_lines
        elif end_line > num_lines:
            warning_message = (
                f"We only show up to {num_lines} since there're only {num_lines} "
                "lines in this file."
            )
            end_line = num_lines

        if end_line < start_line:
            return ToolResult(
                output=(
                    f"Its second element `{end_line}` should be greater than or "
                    f"equal to the first element `{start_line}`."
                ),
                success=False,
            )

        snippet = self._editor_slice_content(file_content, start_line, end_line)
        output = self._editor_make_output(
            "\n".join(snippet.splitlines()), str(path), start_line,
        )
        if warning_message:
            output = f"NOTE: {warning_message}\n{output}"
        return ToolResult(output=output, success=True)

    def _editor_create(self, path: str, file_text: Optional[str]) -> "ToolResult":
        """Create a new file."""

        if file_text is None:
            return ToolResult(
                output="Error: file_text is required for create command",
                success=False,
            )

        result = self.write_file(path, file_text)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to write to {path}",
                success=False,
            )

        self._editor_add_history(path, file_text)

        return ToolResult(
            output=f"File created successfully at: {path}",
            success=True,
        )

    def _editor_str_replace(
        self, path: str, old_str: Optional[str], new_str: Optional[str]
    ) -> "ToolResult":
        """Replace a string in a file."""

        if old_str is None:
            return ToolResult(
                output="Error: old_str is required for str_replace command",
                success=False,
            )
        if new_str is None:
            return ToolResult(
                output="Error: new_str is required for str_replace command",
                success=False,
            )
        if new_str == old_str:
            return ToolResult(
                output=(
                    "No replacement was performed. `new_str` and `old_str` must be "
                    "different."
                ),
                success=False,
            )

        file_error = self._editor_validate_file(path)
        if file_error is not None:
            return file_error

        result = self.read_file(path)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to read {path}",
                success=False,
            )

        file_content = result.output
        new_str = new_str or ""

        pattern = re.escape(old_str)
        occurrences = [
            (
                file_content.count("\n", 0, match.start()) + 1,
                match.group(),
                match.start(),
            )
            for match in re.finditer(pattern, file_content)
        ]

        if not occurrences:
            old_str = old_str.strip()
            new_str = new_str.strip()
            pattern = re.escape(old_str)
            occurrences = [
                (
                    file_content.count("\n", 0, match.start()) + 1,
                    match.group(),
                    match.start(),
                )
                for match in re.finditer(pattern, file_content)
            ]
            if not occurrences:
                return ToolResult(
                    output=(
                        f"No replacement was performed, old_str `{old_str}` did not "
                        f"appear verbatim in {path}."
                    ),
                    success=False,
                )

        if len(occurrences) > 1:
            line_numbers = sorted(set(line for line, _, _ in occurrences))
            return ToolResult(
                output=(
                    "No replacement was performed. Multiple occurrences of old_str "
                    f"`{old_str}` in lines {line_numbers}. Please ensure it is "
                    "unique."
                ),
                success=False,
            )

        replacement_line, matched_text, idx = occurrences[0]
        new_file_content = (
            file_content[:idx] + new_str + file_content[idx + len(matched_text) :]
        )

        result = self.write_file(path, new_file_content)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to write to {path}",
                success=False,
            )

        self._editor_add_history(path, file_content)

        start_line = max(0, replacement_line - self.SNIPPET_CONTEXT_WINDOW)
        end_line = replacement_line + self.SNIPPET_CONTEXT_WINDOW + new_str.count("\n")
        snippet = self._editor_slice_content(
            new_file_content, start_line + 1, end_line
        )

        success_message = f"The file {path} has been edited. "
        success_message += self._editor_make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_message += (
            "Review the changes and make sure they are as expected. Edit the "
            "file again if necessary."
        )

        return ToolResult(
            output=success_message,
            success=True,
        )

    def _editor_insert(
        self, path: str, insert_line: Optional[int], new_str: Optional[str]
    ) -> "ToolResult":
        """Insert text after a specific line."""


        if insert_line is None:
            return ToolResult(
                output="Error: insert_line is required for insert command",
                success=False,
            )

        if new_str is None:
            return ToolResult(
                output="Error: new_str is required for insert command",
                success=False,
            )

        file_error = self._editor_validate_file(path)
        if file_error is not None:
            return file_error

        result = self.read_file(path)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to read {path}",
                success=False,
            )

        file_content = result.output
        num_lines = self._editor_count_lines(file_content)

        if insert_line < 0 or insert_line > num_lines:
            return ToolResult(
                output=(
                    f"It should be within the range of allowed values: "
                    f"{[0, num_lines]}"
                ),
                success=False,
            )

        new_str_lines = new_str.split("\n")
        insert_text = "".join(line + "\n" for line in new_str_lines)
        lines = file_content.splitlines(keepends=True)
        new_content = (
            "".join(lines[:insert_line]) + insert_text + "".join(lines[insert_line:])
        )

        result = self.write_file(path, new_content)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to write to {path}",
                success=False,
            )

        self._editor_add_history(path, file_content)

        start_line = max(0, insert_line - self.SNIPPET_CONTEXT_WINDOW)
        end_line = min(
            num_lines + len(new_str_lines),
            insert_line + self.SNIPPET_CONTEXT_WINDOW + len(new_str_lines),
        )
        snippet = self._editor_slice_content(new_content, start_line + 1, end_line)

        success_message = f"The file {path} has been edited. "
        success_message += self._editor_make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - self.SNIPPET_CONTEXT_WINDOW + 1),
        )
        success_message += (
            "Review the changes and make sure they are as expected (correct "
            "indentation, no duplicate lines, etc). Edit the file again if necessary."
        )

        return ToolResult(
            output=success_message,
            success=True,
        )

    def _editor_undo(self, path: str) -> "ToolResult":
        """Undo the last edit to a file."""

        result = self.read_file(path)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to read {path}",
                success=False,
            )
        old_text = self._editor_pop_history(path)
        if old_text is None:
            return ToolResult(
                output=f"No edit history found for {path}.",
                success=False,
            )

        result = self.write_file(path, old_text)
        if result.exit_code != 0:
            return ToolResult(
                output=f"Ran into {result.output} while trying to write to {path}",
                success=False,
            )

        output = (
            f"Last edit to {path} undone successfully. "
            f"{self._editor_make_output(old_text, str(path))}"
        )
        return ToolResult(output=output, success=True)

    def _truncate_output(self, output: str) -> str:
        """Truncate output and mark with <response clipped> if needed."""
        return maybe_truncate(
            output,
            truncate_after=self.max_output_chars,
            truncate_notice=DEFAULT_TRUNCATE_NOTICE,
        )

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> ExecutionResult:
        """Read a file from the environment."""
        result = self.execute(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            return result
        if start_line is not None and end_line is not None:
            lines = result.output.splitlines(keepends=True)
            sliced = "".join(lines[start_line - 1 : end_line])
            return ExecutionResult(output=sliced, exit_code=0)
        if start_line is not None or end_line is not None:
            return ExecutionResult(
                output="Both start_line and end_line must be provided together",
                exit_code=1,
            )
        return result

    def write_file(self, path: str, content: str) -> ExecutionResult:
        """Write content to a file in the environment."""
        path_quoted = shlex.quote(path)
        cmd = (
            f"cat > {path_quoted} << 'LITE_OPENHANDS_EOF_31415926'\n"
            f"{content}\nLITE_OPENHANDS_EOF_31415926"
        )
        return self.execute(cmd)

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the environment."""
        result = self.execute(f"test -f {shlex.quote(path)}")
        return result.exit_code == 0

    def is_directory(self, path: str) -> bool:
        """Check if a path is a directory."""
        result = self.execute(f"test -d {shlex.quote(path)}")
        return result.exit_code == 0

    def list_directory(self, path: str, max_depth: int = 2) -> str:
        """List directory contents."""
        path_quoted = shlex.quote(path)
        result = self.execute(
            f"find {path_quoted} -maxdepth {max_depth} -not -path '*/.*' "
            "2>/dev/null | head -100"
        )
        return result.output

    def get_git_diff(self) -> str:
        """Get the git diff of changes made in the workspace."""
        repo_dir = getattr(self, "_repo_dir", None) or self.workspace_dir
        base_commit = getattr(self, "_base_commit", None)
        repo_dir = repo_dir.rstrip("/") if repo_dir else repo_dir
        if not repo_dir:
            return ""

        repo_dir_quoted = shlex.quote(repo_dir)
        check = self.execute(
            f"cd {repo_dir_quoted} && git rev-parse --is-inside-work-tree"
        )
        if check.exit_code != 0:
            return ""

        self.execute(f"cd {repo_dir_quoted} && git add -A")
        self.execute(
            f"cd {repo_dir_quoted} && "
            f"git config --global user.email '{GIT_USER_EMAIL}' && "
            f"git config --global user.name '{GIT_USER_NAME}'"
        )
        self.execute(
            f"cd {repo_dir_quoted} && "
            f"git commit --no-verify -m '{GIT_COMMIT_MESSAGE}'"
        )

        if base_commit:
            diff_result = self.execute(
                f"cd {repo_dir_quoted} && "
                f"git --no-pager diff --no-color {base_commit} HEAD"
            )
            if diff_result.exit_code == 0:
                return diff_result.output

        # After commit, working tree is clean so `git diff` would be empty.
        # Use HEAD~1..HEAD to capture the just-committed changes.
        fallback = self.execute(
            f"cd {repo_dir_quoted} && git --no-pager diff --no-color HEAD~1 HEAD"
        )
        return fallback.output

    def undo_edit(self, path: str) -> ExecutionResult:
        """Undo the last edit to a file. Default returns error."""
        return ExecutionResult(
            output="Undo not supported by this environment", exit_code=1
        )

    def __enter__(self) -> "ShellEnvironment":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


BaseEnvironment = ShellEnvironment
