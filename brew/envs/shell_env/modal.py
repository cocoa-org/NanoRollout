"""
Modal-based environment implementation.
"""
import asyncio
import logging
import os
import re
import shlex
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .base import ExecutionResult, ShellEnvironment, extract_cwd_marker

if TYPE_CHECKING:
    from modal import App, Image, Sandbox
else:
    App = Image = Sandbox = Secret = Volume = None

logger = logging.getLogger(__name__)

_PWD_MARKER = "__TINYBREW_PWD__"
_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_modal_symbols() -> None:
    global App, Image, Sandbox, Secret, Volume

    if all(symbol is not None for symbol in (App, Image, Sandbox, Secret, Volume)):
        return

    from modal import App as ModalApp
    from modal import Image as ModalImage
    from modal import Sandbox as ModalSandbox
    from modal import Secret as ModalSecret
    from modal import Volume as ModalVolume

    App = ModalApp
    Image = ModalImage
    Sandbox = ModalSandbox
    Secret = ModalSecret
    Volume = ModalVolume


@dataclass
class ModalEnvironmentConfig:
    image: str
    cwd: str = "/workspace"
    timeout: int = 120
    cpu: float | None = 0.5
    memory_mb: int | None = 128
    gpu: str | None = None
    allow_internet: bool = True
    app_name: str = "__tinybrew__"
    sandbox_name: str | None = None
    sandbox_timeout: int = 24 * 60 * 60
    dockerfile_name: str = "Dockerfile"
    env: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    volumes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ModalEnvironment(ShellEnvironment):
    """
    A Modal Sandbox-based execution environment.

    The image can be either:
    - a container registry image reference, or
    - a local directory containing a Dockerfile.
    """

    def __init__(
        self,
        image: str,
        instance: dict[str, Any] | None = None,
        workspace_dir: str = "/workspace",
        timeout: int = 120,
        env: Optional[dict[str, str]] = None,
        cpu: float | None = 0.5,
        memory_mb: int | None = 128,
        gpu: str | None = None,
        allow_internet: bool = True,
        app_name: str = "__tinybrew__",
        sandbox_name: str | None = None,
        sandbox_timeout: int = 24 * 60 * 60,
        dockerfile_name: str = "Dockerfile",
        secrets: Optional[list[str]] = None,
        volumes: Optional[dict[str, str]] = None,
        logger_override: Optional[logging.Logger] = None,
    ):
        self.logger = logger_override or logger
        self.instance = instance or {}
        self.config = ModalEnvironmentConfig(
            image=image,
            cwd=workspace_dir,
            timeout=timeout,
            env=env or {},
            cpu=cpu,
            memory_mb=memory_mb,
            gpu=gpu,
            allow_internet=allow_internet,
            app_name=app_name,
            sandbox_name=sandbox_name,
            sandbox_timeout=sandbox_timeout,
            dockerfile_name=dockerfile_name,
            secrets=secrets or [],
            volumes=volumes or {},
        )

        invalid_env_names = sorted(
            name
            for name in self.config.env
            if not _ENV_VAR_NAME_PATTERN.match(name)
        )
        if invalid_env_names:
            raise ValueError(
                "Invalid environment variable names: "
                + ", ".join(invalid_env_names)
            )

        self.workspace_dir = self.config.cwd
        self.timeout = self.config.timeout
        self._cwd = self.config.cwd

        self._app: "App | None" = None
        self._image: "Image | None" = None
        self._sandbox: "Sandbox | None" = None

    def start(self) -> None:
        if self._sandbox is not None:
            return
        self._run_async(self._start_async())

    async def _start_async(self) -> None:
        _ensure_modal_symbols()
        self._image = self._build_image()
        assert App is not None
        assert Sandbox is not None
        assert Secret is not None
        assert Volume is not None

        self._app = await App.lookup.aio(
            name=self.config.app_name,
            create_if_missing=True,
        )

        sandbox_kwargs: dict[str, Any] = {
            "app": self._app,
            "image": self._image,
            "name": self._resolve_sandbox_name(),
            "timeout": self.config.sandbox_timeout,
            "block_network": not self.config.allow_internet,
        }
        if self.config.cpu is not None:
            sandbox_kwargs["cpu"] = self.config.cpu
        if self.config.memory_mb is not None:
            sandbox_kwargs["memory"] = self.config.memory_mb
        if self.config.gpu is not None:
            sandbox_kwargs["gpu"] = self.config.gpu
        if self.config.secrets:
            sandbox_kwargs["secrets"] = [
                Secret.from_name(secret_name) for secret_name in self.config.secrets
            ]
        if self.config.volumes:
            sandbox_kwargs["volumes"] = {
                mount_path: Volume.from_name(volume_name)
                for mount_path, volume_name in self.config.volumes.items()
            }

        self._sandbox = await Sandbox.create.aio(**sandbox_kwargs)

        mkdir_command = f"mkdir -p {shlex.quote(self.workspace_dir)}"
        _, stderr, exit_code = await self._exec_process_async(
            mkdir_command,
            timeout_sec=self.timeout,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to create workspace {self.workspace_dir}: {stderr}"
            )
        self._cwd = self.workspace_dir

    def stop(self) -> None:
        if self._sandbox is None:
            return
        self._run_async(self._stop_async())

    async def _stop_async(self) -> None:
        sandbox = self._sandbox
        self._sandbox = None
        self._app = None
        self._image = None
        if sandbox is None:
            return

        try:
            await sandbox.terminate.aio()
            await sandbox.wait.aio(raise_on_termination=False)
        except Exception as exc:
            self.logger.warning("Error terminating Modal sandbox: %s", exc)

    def is_running(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            _, _, exit_code = self._run_async(
                self._exec_process_async("true", timeout_sec=5)
            )
            return exit_code == 0
        except Exception:
            return False

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecutionResult:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")

        export_lines = "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in self.config.env.items()
        )

        wrapped_command = (
            f"cd {shlex.quote(self._cwd)}\n"
            f"{export_lines}"
            "exec 2>&1\n"
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{_PWD_MARKER}%s\\n' \"$(pwd)\"\n"
            "exit $status\n"
        )

        try:
            stdout, stderr, exit_code = self._run_async(
                self._exec_process_async(
                    wrapped_command,
                    timeout_sec=self.timeout if timeout is None else timeout,
                )
            )
        except TimeoutError as exc:
            return ExecutionResult(
                output=str(exc) or f"Command timed out after {self.timeout if timeout is None else timeout}s",
                exit_code=124,
            )
        except Exception as exc:
            return ExecutionResult(output=f"Error: {exc}", exit_code=1)

        output, new_cwd = extract_cwd_marker(stdout, _PWD_MARKER)
        if new_cwd:
            self._cwd = new_cwd
        if stderr:
            output = f"{output}{stderr}"

        return ExecutionResult(output=output, exit_code=exit_code)

    def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self._run_async(self._upload_file_async(Path(source_path), target_path))

    async def _upload_file_async(self, source_path: Path, target_path: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        async with await self._sandbox.open.aio(target_path, "wb") as remote_file:
            with open(source_path, "rb") as local_file:
                while True:
                    chunk = local_file.read(8192)
                    if not chunk:
                        break
                    await remote_file.write.aio(chunk)

    def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self._run_async(self._upload_dir_async(Path(source_dir), target_dir))

    async def _upload_dir_async(self, source_dir: Path, target_dir: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")

        await self._exec_process_async(
            f"mkdir -p {shlex.quote(target_dir)}",
            timeout_sec=self.timeout,
        )
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(source_dir)
            remote_file_path = str(Path(target_dir) / relative_path)
            remote_parent = str(Path(remote_file_path).parent)
            await self._exec_process_async(
                f"mkdir -p {shlex.quote(remote_parent)}",
                timeout_sec=self.timeout,
            )
            await self._upload_file_async(file_path, remote_file_path)

    def download_file(self, source_path: str, target_path: Path | str) -> None:
        self._run_async(self._download_file_async(source_path, Path(target_path)))

    async def _download_file_async(self, source_path: str, target_path: Path) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        async with await self._sandbox.open.aio(source_path, "rb") as remote_file:
            with open(target_path, "wb") as local_file:
                while True:
                    chunk = await remote_file.read.aio(8192)
                    if not chunk:
                        break
                    local_file.write(chunk)

    def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        self._run_async(self._download_dir_async(source_dir, Path(target_dir)))

    async def _download_dir_async(self, source_dir: str, target_dir: Path) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        target_dir.mkdir(parents=True, exist_ok=True)

        children = await self._sandbox.ls.aio(source_dir)
        for child_name in children:
            child_name_str = str(child_name)
            child_path = str(Path(source_dir) / child_name_str)
            local_path = target_dir / child_name_str
            try:
                await self._sandbox.ls.aio(child_path)
            except NotADirectoryError:
                await self._download_file_async(child_path, local_path)
            else:
                await self._download_dir_async(child_path, local_path)

    def attach(self) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")
        os.execvp(
            "modal",
            [
                "modal",
                "shell",
                str(self._sandbox.object_id),
            ],
        )

    def _build_image(self) -> "Image":
        _ensure_modal_symbols()
        assert Image is not None
        image_path = Path(self.config.image).expanduser()
        if image_path.exists():
            if image_path.is_dir():
                dockerfile_path = image_path / self.config.dockerfile_name
                context_dir = image_path
            else:
                dockerfile_path = image_path
                context_dir = image_path.parent
            if dockerfile_path.exists():
                return Image.from_dockerfile(
                    str(dockerfile_path),
                    context_dir=str(context_dir),
                )

        return Image.from_registry(self.config.image)

    async def _exec_process_async(
        self,
        command: str,
        timeout_sec: Optional[int] = None,
    ) -> tuple[str, str, int]:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")

        process = await self._sandbox.exec.aio(
            "bash",
            "-lc",
            command,
            timeout=timeout_sec,
        )
        stdout = self._normalize_stream_data(await process.stdout.read.aio())
        stderr = self._normalize_stream_data(await process.stderr.read.aio())
        return_code = await process.wait.aio()
        return stdout, stderr, self._normalize_exit_code(return_code)

    def _resolve_sandbox_name(self) -> str:
        if self.config.sandbox_name:
            return self.config.sandbox_name

        instance_id = str(self.instance.get("instance_id", "task"))
        safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "-", instance_id).strip("-")
        if not safe_instance:
            safe_instance = "task"
        return f"tinybrew-{safe_instance}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _run_async(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "ModalEnvironment provides synchronous APIs and cannot run inside an "
            "already running event loop."
        )

    @staticmethod
    def _normalize_stream_data(stream_data: Any) -> str:
        if stream_data is None:
            return ""
        if isinstance(stream_data, bytes):
            return stream_data.decode("utf-8", errors="replace")
        return str(stream_data)

    @staticmethod
    def _normalize_exit_code(exit_code: Any) -> int:
        try:
            return int(exit_code)
        except (TypeError, ValueError):
            return 1
