"""
Docker-based environment implementation.

This module provides a Docker container-based execution environment.
"""

import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from docker.models.containers import Container

from .base import ExecutionResult, ShellEnvironment, extract_cwd_marker


class DockerEnvironment(ShellEnvironment):
    """
    A Docker container-based execution environment.

    This environment runs commands inside a Docker container and provides
    file editing capabilities with undo support.
    """

    def __init__(
        self,
        image: str,
        workspace_dir: str = "/workspace",
        timeout: int = 120,
    ):
        """
        Initialize the Docker environment.

        Args:
            image: Docker image to use
            workspace_dir: Working directory inside the container
            timeout: Default command timeout in seconds
        """
        self.image = image
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self.client = self._load_docker_client()
        self.container: Optional["Container"] = None
        self._file_history: dict[str, list[str]] = {}  # For undo support
        self._cwd = workspace_dir

    @staticmethod
    def _load_docker_client() -> Any:
        import docker

        return docker.from_env()

    def start(self) -> None:
        """Start the Docker container."""
        self.container = self.client.containers.run(
            self.image,
            command="/bin/bash",
            stdin_open=True,
            tty=True,
            detach=True,
            working_dir=self.workspace_dir,
            platform="linux/amd64",
        )
        # Wait for container to be ready
        time.sleep(1)

    def stop(self) -> None:
        """Stop and remove the Docker container."""
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def is_running(self) -> bool:
        """Check if the container is running."""
        if not self.container:
            return False
        try:
            self.container.reload()
            return self.container.status == "running"
        except Exception:
            return False

    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute a bash command in the container.

        Args:
            command: The bash command to execute
            timeout: Optional timeout in seconds

        Returns:
            ExecutionResult with output and exit code
        """
        if not self.container:
            raise RuntimeError("Container not started")

        marker = "__TINYBREW_PWD__"
        wrapped_command = (
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{marker}%s\\n' \"$(pwd)\"\n"
            "exit $status"
        )
        exec_command: list[str] = ["/bin/bash", "-lc", wrapped_command]
        if timeout is not None:
            exec_command = ["timeout", f"{timeout}s", *exec_command]

        try:
            exit_code, output = self.container.exec_run(
                exec_command,
                workdir=self._cwd,
                demux=False,
            )
            output_str = output.decode("utf-8", errors="replace") if output else ""
            output_str, new_cwd = extract_cwd_marker(output_str, marker)
            if new_cwd:
                self._cwd = new_cwd
            return ExecutionResult(output=output_str, exit_code=exit_code)
        except Exception as e:
            return ExecutionResult(output=f"Error: {str(e)}", exit_code=1)

    def write_file(self, path: str, content: str) -> ExecutionResult:
        """Write content to a file in the container with undo support."""
        # Save current content for undo
        current = self.execute(f"cat {repr(path)} 2>/dev/null")
        if current.exit_code == 0:
            if path not in self._file_history:
                self._file_history[path] = []
            self._file_history[path].append(current.output)

        # Use a heredoc to write the file
        cmd = f"cat > {repr(path)} << 'TINYBREW_EOF'\n{content}\nTINYBREW_EOF"
        return self.execute(cmd)

    def undo_edit(self, path: str) -> ExecutionResult:
        """Undo the last edit to a file."""
        if path not in self._file_history or not self._file_history[path]:
            return ExecutionResult(output="No edit history for this file", exit_code=1)

        previous_content = self._file_history[path].pop()
        # Write without saving to history (avoid infinite loop)
        cmd = f"cat > {repr(path)} << 'TINYBREW_EOF'\n{previous_content}\nTINYBREW_EOF"
        return self.execute(cmd)
