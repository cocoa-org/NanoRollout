"""
Enroot-based environment implementation.

Will deprecate soon and use Apptainer/Singularity instead.
"""
import errno
import logging
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base import ExecutionResult, ShellEnvironment, extract_cwd_marker

logger = logging.getLogger(__name__)


@dataclass
class EnrootEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    timeout: int = 180  # Default timeout for command execution in seconds
    executable: str = "enroot"  # Enroot executable path
    create_args: list[str] = field(default_factory=list)
    start_args: list[str] = field(default_factory=lambda: ["--rw", "--root"])
    create_timeout: int = 1200
    step_timeout: int = 600
    eval_timeout: int = 600

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EnrootEnvironment(ShellEnvironment):
    """
    An Enroot container-based execution environment.
    """

    def __init__(
        self,
        image: str,
        instance: dict[str, Any],
        workspace_dir: str = "/",
        timeout: int = 120,
        env: Optional[dict[str, str]] = None,
        forward_env: Optional[list[str]] = None,
        executable: str = "enroot",
        create_args: Optional[list[str]] = None,
        start_args: Optional[list[str]] = None,
        create_timeout: int = 600,
        step_timeout: int = 600,
        eval_timeout: int = 600,
        logger_override: Optional[logging.Logger] = None,
    ):
        self.logger = logger_override or logger
        self.instance = instance
        self.config = EnrootEnvironmentConfig(
            image=image,
            cwd=workspace_dir,
            env=env or {},
            forward_env=forward_env or [],
            timeout=timeout,
            executable=executable,
            create_args=create_args or [],
            start_args=start_args or ["--rw", "--root"],
            create_timeout=create_timeout,
            step_timeout=step_timeout,
            eval_timeout=eval_timeout,
        )
        self.workspace_dir = self.config.cwd
        self.timeout = self.config.timeout
        self._cwd = self.config.cwd
        self.container_name: Optional[str] = None
        self.container_pid: Optional[int] = None
        self.container_process: Optional[subprocess.Popen[str]] = None
        self._orphan_reaper_process: Optional[subprocess.Popen[str]] = None
        self._file_history: dict[str, list[str]] = {}
        self._resolve_image_path()

    def _resolve_image_path(self) -> None:
        if "://" in self.config.image:
            return

        image_path = Path(self.config.image).expanduser()
        if image_path.exists():
            self.config.image = str(image_path.resolve())
            return

        cache_paths: list[Path] = []
        cache_env = os.environ.get("ENROOT_CACHE_PATH", "")
        if cache_env.strip():
            for entry in cache_env.split(":"):
                if entry:
                    cache_paths.append(Path(entry).expanduser())
        else:
            cache_paths.extend(
                [
                    Path.home() / ".cache" / "enroot",
                    Path("/var/cache/enroot"),
                ]
            )

        candidate_names = [self.config.image]
        if image_path.suffix == "":
            candidate_names.append(f"{self.config.image}.sqsh")

        for cache_dir in cache_paths:
            for name in candidate_names:
                candidate = cache_dir / name
                if candidate.exists():
                    self.logger.info(
                        "Resolved image %s to %s", self.config.image, candidate
                    )
                    self.config.image = str(candidate.resolve())
                    return

        self.logger.error(
            "Image %s not found in any cache directory", self.config.image
        )
        raise FileNotFoundError(
            f"Image {self.config.image} not found in any cache directory"
        )

    def start(self) -> None:
        """Create and start the Enroot container."""
        self.container_name = f"{self.instance['instance_id']}-{uuid.uuid4().hex[:8]}"
        self.logger.info(
            "Starting container %s for instance %s",
            self.container_name,
            self.instance['instance_id'],
        )
        create_cmd = [
            self.config.executable,
            "create",
            "--name",
            self.container_name,
            *self.config.create_args,
            self.config.image,
        ]
        self.logger.debug("Creating container with command: %s", shlex.join(create_cmd))
        result = subprocess.run(
            create_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=self.config.create_timeout if "xarray" not in self.config.image else 1800,
        )
        self.logger.info(
            "Created container %s: %s", self.container_name, result.stdout.strip()
        )

        start_cmd = [
            self.config.executable,
            "start",
            *self.config.start_args,
            self.container_name,
            "sh",
            "-c",
            "echo $$; exec sleep infinity",
        ]

        self.logger.debug("Starting container with command: %s", shlex.join(start_cmd))
        self.container_process = subprocess.Popen(
            start_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )

        try:
            assert self.container_process.stdout is not None
            start_time = time.time()
            pid_str = ""
            while time.time() - start_time < 10:
                line = self.container_process.stdout.readline()
                if line:
                    pid_str = line.strip()
                    break
                if self.container_process.poll() is not None:
                    break
                time.sleep(0.1)

            if not pid_str or not pid_str.isdigit():
                stderr_out = ""
                if self.container_process.stderr:
                    stderr_out = self.container_process.stderr.read()
                raise RuntimeError(
                    "Failed to get PID from container. "
                    f"stdout: '{pid_str}', stderr: '{stderr_out}'"
                )

            self.container_pid = int(pid_str)
            self.logger.info(
                "Started container %s with PID %d",
                self.container_name,
                self.container_pid,
            )
            self._start_orphan_reaper()
        except Exception as exc:
            self.logger.error("Failed to start container process: %s", exc)
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the background process and remove the container."""
        self._stop_orphan_reaper()

        if self.container_process is not None:
            self.logger.info("Stopping container process...")
            self._terminate_process_group(self.container_process, term_timeout_s=5.0)
            self.container_process = None
            self.container_pid = None

        if self.container_name is not None:
            cmd = [self.config.executable, "remove", "-f", self.container_name]
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            except (subprocess.SubprocessError, OSError):
                # Already cleaned up
                pass
            self.container_name = None

    def is_running(self) -> bool:
        """Check if the container is running."""
        if self.container_process is None or self.container_pid is None:
            return False
        if self.container_process.poll() is not None:
            return False
        try:
            os.kill(self.container_pid, 0)
        except OSError:
            return False
        return True

    def _terminate_process_group(
        self, process: subprocess.Popen[str], *, term_timeout_s: float = 5.0
    ) -> None:
        """Best-effort terminate for a process and all descendants in its process group."""
        if process.poll() is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            return
        except Exception as exc:
            self.logger.warning("Error resolving process group for pid=%s: %s", process.pid, exc)
            return

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as exc:
            self.logger.warning("Error sending SIGTERM to process group %s: %s", pgid, exc)

        try:
            process.wait(timeout=term_timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception as exc:
            self.logger.warning("Error sending SIGKILL to process group %s: %s", pgid, exc)
            return

        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Timed out waiting for process group %s to exit after SIGKILL.",
                pgid,
            )

    def _start_orphan_reaper(self) -> None:
        """Spawn a detached watcher to cleanup enroot resources if this parent process dies."""
        self._stop_orphan_reaper()
        if (
            self.container_name is None
            or self.container_pid is None
            or self.container_process is None
        ):
            return

        # NOTE: This process is intentionally detached from the ray worker process group.
        # If the worker is force-killed, this watcher can still remove leftover enroot
        # processes/containers that would otherwise become orphaned.
        parent_pid = os.getpid()
        start_process_pid = self.container_process.pid
        container_pid = self.container_pid
        container_name = self.container_name
        executable = self.config.executable

        reaper_script = textwrap.dedent(
            r"""
            import os
            import signal
            import subprocess
            import sys
            import time

            parent_pid = int(sys.argv[1])
            start_process_pid = int(sys.argv[2])
            container_pid = int(sys.argv[3])
            container_name = sys.argv[4]
            executable = sys.argv[5]

            def _parent_alive() -> bool:
                try:
                    os.kill(parent_pid, 0)
                    return True
                except OSError:
                    return False

            def _read_cmdline(pid: int) -> str:
                path = os.path.join("/proc", str(pid), "cmdline")
                try:
                    with open(path, "rb") as handle:
                        raw = handle.read()
                except Exception:
                    return ""
                if not raw:
                    return ""
                return raw.decode("utf-8", "ignore").replace("\x00", " ")

            def _looks_like_our_start_process() -> bool:
                cmdline = _read_cmdline(start_process_pid)
                if not cmdline:
                    return False
                return (
                    "enroot" in cmdline
                    and " start " in f" {cmdline} "
                    and container_name in cmdline
                )

            def _kill_group(sig: int) -> None:
                if not _looks_like_our_start_process():
                    return
                try:
                    os.killpg(start_process_pid, sig)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass

            def _kill_matching_exec(sig: int) -> None:
                needle_pid = f"enroot exec {container_pid} "
                needle_name = f"enroot exec {container_name} "
                marker = "__NANOROLLOUT_PWD__"
                proc_root = "/proc"
                if not os.path.isdir(proc_root):
                    return
                for name in os.listdir(proc_root):
                    if not name.isdigit():
                        continue
                    pid = int(name)
                    if pid == os.getpid():
                        continue
                    cmdline_path = os.path.join(proc_root, name, "cmdline")
                    try:
                        with open(cmdline_path, "rb") as handle:
                            raw = handle.read()
                    except Exception:
                        continue
                    if not raw:
                        continue
                    cmdline = raw.decode("utf-8", "ignore").replace("\x00", " ")
                    if needle_pid not in cmdline and needle_name not in cmdline:
                        continue
                    # Narrow down to NanoRollout-launched enroot exec commands only.
                    if marker not in cmdline:
                        continue
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        pass

            while _parent_alive():
                time.sleep(2.0)

            _kill_group(signal.SIGTERM)
            _kill_matching_exec(signal.SIGTERM)
            time.sleep(1.0)
            _kill_group(signal.SIGKILL)
            _kill_matching_exec(signal.SIGKILL)

            try:
                subprocess.run(
                    [executable, "remove", "-f", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=30,
                )
            except Exception:
                pass
            """
        )
        python_bin = sys.executable or "python3"
        try:
            self._orphan_reaper_process = subprocess.Popen(
                [
                    python_bin,
                    "-c",
                    reaper_script,
                    str(parent_pid),
                    str(start_process_pid),
                    str(container_pid),
                    container_name,
                    executable,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                text=False,
            )
            self.logger.debug(
                "Started orphan reaper pid=%s for container=%s",
                self._orphan_reaper_process.pid,
                container_name,
            )
        except Exception as exc:
            self._orphan_reaper_process = None
            self.logger.warning("Failed to start orphan reaper: %s", exc)

    def _stop_orphan_reaper(self) -> None:
        proc = self._orphan_reaper_process
        if proc is None:
            return
        self._orphan_reaper_process = None
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_subprocess(
        self,
        cmd: list[str],
        *,
        timeout_s: int,
        stdin_data: Optional[str] = None,
    ) -> ExecutionResult:
        """Run subprocess with robust timeout cleanup of descendant processes."""
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(input=stdin_data, timeout=timeout_s)
            return ExecutionResult(
                output=stdout or "",
                exit_code=process.returncode if process.returncode is not None else 1,
            )
        except subprocess.TimeoutExpired as exc:
            self.logger.warning(
                "Command timed out after %ss; terminating process group. cmd=%s",
                timeout_s,
                shlex.join(cmd),
            )
            self._terminate_process_group(process, term_timeout_s=5.0)
            partial_output = exc.stdout or ""
            timeout_msg = f"Command timed out after {timeout_s}s"
            output = (
                f"{partial_output.rstrip()}\n{timeout_msg}"
                if partial_output
                else timeout_msg
            )
            return ExecutionResult(output=output, exit_code=124)

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Execute a bash command in the container."""
        if self.container_pid is None:
            raise RuntimeError("Container is not running or PID is unknown")

        marker = "__NANOROLLOUT_PWD__"
        wrapped_command = (
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{marker}%s\\n' \"$(pwd)\"\n"
            "exit $status"
        )
        safe_cwd = shlex.quote(self._cwd)
        full_command = f"cd {safe_cwd} && {wrapped_command}"

        base_cmd = [self.config.executable, "exec"]

        for key in self.config.forward_env:
            value = os.getenv(key)
            if value is not None:
                base_cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            base_cmd.extend(["--env", f"{key}={value}"])

        timeout_s = self.config.timeout if timeout is None else timeout
        cmd = [*base_cmd, str(self.container_pid), "bash", "-lc", full_command]
        try:
            result = self._run_subprocess(cmd, timeout_s=timeout_s)
        except OSError as exc:
            if exc.errno != errno.E2BIG:
                raise
            # Fall back to stdin to avoid argv size limits for large commands.
            stdin_cmd = [*base_cmd, str(self.container_pid), "bash", "-l", "-s"]
            result = self._run_subprocess(
                stdin_cmd,
                timeout_s=timeout_s,
                stdin_data=full_command,
            )

        output_str = result.output or ""
        output_str, new_cwd = extract_cwd_marker(output_str, marker)
        if new_cwd:
            self._cwd = new_cwd

        return ExecutionResult(output=output_str, exit_code=result.exit_code)

    def write_file(self, path: str, content: str) -> ExecutionResult:
        """Write content to a file in the container with undo support."""
        current = self.execute(f"cat {repr(path)} 2>/dev/null")
        if current.exit_code == 0:
            if path not in self._file_history:
                self._file_history[path] = []
            self._file_history[path].append(current.output)

        cmd = (
            f"cat > {repr(path)} << 'NANOROLLOUT_EOF'\n"
            f"{content}\nNANOROLLOUT_EOF"
        )
        return self.execute(cmd)

    def undo_edit(self, path: str) -> ExecutionResult:
        """Undo the last edit to a file."""
        if path not in self._file_history or not self._file_history[path]:
            return ExecutionResult(output="No edit history for this file", exit_code=1)

        previous_content = self._file_history[path].pop()
        cmd = (
            f"cat > {repr(path)} << 'NANOROLLOUT_EOF'\n"
            f"{previous_content}\nNANOROLLOUT_EOF"
        )
        return self.execute(cmd)

    def __del__(self) -> None:
        self.stop()
