"""GCE-backed remote execution environment.

Mirrors ModalEnvironment's surface so callers can swap backends without
changing their training loop. Implementation: gRPC to a WorkerService
daemon on each worker VM, with VM lifecycle mediated by PoolCoordinator.
Sync exterior, async internals (mirrors modal.py because
BaseEnvironment.execute is abstract sync)."""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Any, Optional

import grpc

from ._pool import PoolCoordinator, VMInfo
from .base import BaseEnvironment, ExecutionResult, extract_cwd_marker
from .gce_worker.proto import worker_pb2, worker_pb2_grpc

logger = logging.getLogger(__name__)

_PWD_MARKER = "__NANOROLLOUT_PWD__"


class GCEEnvironment(BaseEnvironment):
    """Remote container execution via a worker daemon on a GCE VM."""

    def __init__(
        self,
        image: str,
        pool: PoolCoordinator,
        workspace_dir: str = "/workspace",
        timeout: int = 120,
        env: Optional[dict[str, str]] = None,
        cpu_limit: int = 0,
        memory_limit_mb: int = 0,
        worker_port: int = 50051,
        logger_override: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger_override or logger
        self._pool = pool
        self.image = image
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self._env: dict[str, str] = dict(env or {})
        self._cpu_limit = cpu_limit
        self._memory_limit_mb = memory_limit_mb
        self._worker_port = worker_port

        self._cwd = workspace_dir
        self._vm: Optional[VMInfo] = None
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[worker_pb2_grpc.WorkerServiceStub] = None
        self._container_id: Optional[str] = None
        # Persistent event loop. grpc.aio.Channel is bound to its creating
        # loop; using asyncio.run() per call would bind the channel to a
        # loop that's then closed, leading to "coroutine was never awaited"
        # warnings on the next call. We own one loop per env instance and
        # reuse it across start/execute/stop.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ----- BaseEnvironment abstracts (sync exterior) -----

    def start(self) -> None:
        if self._container_id is not None:
            return
        self._run_async(self._start_async())

    def stop(self) -> None:
        if self._container_id is None and self._channel is None:
            # Loop may still need closing if the env was constructed but
            # never .start()'d successfully.
            self._close_loop()
            return
        self._run_async(self._stop_async())
        self._close_loop()

    def is_running(self) -> bool:
        if self._container_id is None or self._stub is None:
            return False
        try:
            result = self._run_async(self._exec_async("true", timeout=10))
            return result.exit_code == 0
        except Exception:
            return False

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecutionResult:
        if self._container_id is None or self._stub is None:
            raise RuntimeError("GCEEnvironment not started")
        actual_timeout = self.timeout if timeout is None else timeout
        return self._run_async(self._exec_async(command, timeout=actual_timeout))

    # ----- file ops (mirroring modal.py upload_file/download_file) -----

    def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self._run_async(self._upload_file_async(Path(source_path), target_path))

    def download_file(self, source_path: str, target_path: Path | str) -> None:
        self._run_async(self._download_file_async(source_path, Path(target_path)))

    # ----- async internals -----

    async def _start_async(self) -> None:
        self._vm = await self._pool.claim()
        target = f"{self._vm.internal_ip}:{self._worker_port}"
        self.logger.info("GCE start: claimed vm=%s target=%s", self._vm.name, target)

        self._channel = grpc.aio.insecure_channel(target)
        self._stub = worker_pb2_grpc.WorkerServiceStub(self._channel)

        try:
            response = await self._stub.StartContainer(
                worker_pb2.StartContainerRequest(
                    image=self.image,
                    work_dir=self.workspace_dir,
                    env_vars={},  # env vars are exported per-execute, like modal.py
                    cpu_limit=self._cpu_limit,
                    memory_limit_mb=self._memory_limit_mb,
                )
            )
        except Exception:
            await self._channel.close()
            self._channel = None
            self._stub = None
            await self._pool.release(self._vm)
            self._vm = None
            raise

        self._container_id = response.container_id
        self._cwd = self.workspace_dir
        self.logger.info("GCE start: container=%s", self._container_id[:12])

    async def _stop_async(self) -> None:
        container_id = self._container_id
        stub = self._stub
        channel = self._channel
        vm = self._vm

        self._container_id = None
        self._stub = None
        self._channel = None
        self._vm = None

        if stub is not None and container_id is not None:
            try:
                await stub.StopContainer(
                    worker_pb2.StopContainerRequest(container_id=container_id)
                )
            except Exception as exc:
                self.logger.warning("GCE stop: StopContainer failed: %s", exc)

        if channel is not None:
            try:
                await channel.close()
            except Exception as exc:
                self.logger.warning("GCE stop: channel close failed: %s", exc)

        if vm is not None:
            try:
                await self._pool.release(vm)
            except Exception as exc:
                self.logger.warning("GCE stop: pool release failed: %s", exc)

    async def _exec_async(
        self,
        command: str,
        timeout: int,
    ) -> ExecutionResult:
        assert self._stub is not None and self._container_id is not None

        wrapped_command = self._wrap_command(command)
        request = worker_pb2.ExecRequest(
            container_id=self._container_id,
            cmd=wrapped_command,
            timeout_sec=int(timeout),
        )

        stdout_buf = bytearray()
        stderr_buf = bytearray()
        exit_code: Optional[int] = None

        # Add a small RPC-level deadline beyond the in-container timeout so
        # the gRPC stream itself can't hang if the container or network
        # silently dies. +30s is generous but still bounded.
        rpc_deadline = float(timeout) + 30.0

        try:
            async for chunk in self._stub.Exec(request, timeout=rpc_deadline):
                if chunk.stream == worker_pb2.ExecChunk.STDOUT:
                    stdout_buf += chunk.data
                elif chunk.stream == worker_pb2.ExecChunk.STDERR:
                    stderr_buf += chunk.data
                elif chunk.stream == worker_pb2.ExecChunk.EXIT:
                    exit_code = int(chunk.exit_code)
        except grpc.aio.AioRpcError as exc:
            if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return ExecutionResult(
                    output=f"Command timed out after {timeout}s",
                    exit_code=124,
                )
            return ExecutionResult(
                output=f"gRPC error: {exc.code().name}: {exc.details()}",
                exit_code=1,
            )

        output = (bytes(stdout_buf) + bytes(stderr_buf)).decode("utf-8", errors="replace")

        if exit_code is None:
            return ExecutionResult(
                output=output + "\n[exec stream ended without EXIT chunk]",
                exit_code=1,
            )

        output, new_cwd = extract_cwd_marker(output, _PWD_MARKER)
        if new_cwd:
            self._cwd = new_cwd

        return ExecutionResult(output=output, exit_code=exit_code)

    async def _upload_file_async(self, source_path: Path, target_path: str) -> None:
        assert self._stub is not None and self._container_id is not None
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        content = source_path.read_bytes()
        await self._stub.WriteFile(
            worker_pb2.WriteFileRequest(
                container_id=self._container_id,
                path=target_path,
                content=content,
            )
        )

    async def _download_file_async(self, source_path: str, target_path: Path) -> None:
        assert self._stub is not None and self._container_id is not None
        response = await self._stub.ReadFile(
            worker_pb2.ReadFileRequest(
                container_id=self._container_id,
                path=source_path,
            )
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)

    # ----- helpers -----

    def _wrap_command(self, command: str) -> str:
        export_lines = "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in self._env.items()
        )
        return (
            f"cd {shlex.quote(self._cwd)}\n"
            f"{export_lines}"
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{_PWD_MARKER}%s\\n' \"$(pwd)\"\n"
            "exit $status\n"
        )

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _close_loop(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception as exc:
                self.logger.warning("GCE close: loop close failed: %s", exc)
        self._loop = None

    def _run_async(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._ensure_loop().run_until_complete(coro)
        raise RuntimeError(
            "GCEEnvironment provides synchronous APIs and cannot run inside "
            "an already running event loop."
        )
