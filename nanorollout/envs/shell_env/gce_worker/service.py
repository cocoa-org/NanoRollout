"""WorkerService gRPC handler implementation.

Runs on grpc.aio.Server. Docker SDK is sync, so blocking calls dispatch
to threads via loop.run_in_executor to keep the asyncio loop responsive."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import grpc

from .docker_runtime import (
    STREAM_EXIT,
    STREAM_STDERR,
    STREAM_STDOUT,
    DockerRuntime,
)
from .proto import worker_pb2, worker_pb2_grpc

WORKER_VERSION = "0.1.0"

logger = logging.getLogger(__name__)


class WorkerService(worker_pb2_grpc.WorkerServiceServicer):
    def __init__(self, runtime: DockerRuntime | None = None) -> None:
        self._runtime = runtime or DockerRuntime()

    # ----- lifecycle -----

    async def StartContainer(
        self,
        request: worker_pb2.StartContainerRequest,
        context: grpc.aio.ServicerContext,
    ) -> worker_pb2.StartContainerResponse:
        loop = asyncio.get_running_loop()
        try:
            container_id = await loop.run_in_executor(
                None,
                lambda: self._runtime.start_container(
                    image=request.image,
                    work_dir=request.work_dir,
                    env_vars=dict(request.env_vars),
                    cpu_limit=request.cpu_limit,
                    memory_limit_mb=request.memory_limit_mb,
                ),
            )
        except Exception as exc:
            logger.exception("StartContainer failed: image=%s", request.image)
            await context.abort(grpc.StatusCode.INTERNAL, f"start_container: {exc}")
            raise  # unreachable
        return worker_pb2.StartContainerResponse(container_id=container_id)

    async def StopContainer(
        self,
        request: worker_pb2.StopContainerRequest,
        context: grpc.aio.ServicerContext,
    ) -> worker_pb2.Empty:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, self._runtime.stop_container, request.container_id
            )
        except Exception as exc:
            logger.exception("StopContainer failed: id=%s", request.container_id[:12])
            await context.abort(grpc.StatusCode.INTERNAL, f"stop_container: {exc}")
            raise
        return worker_pb2.Empty()

    # ----- exec -----

    async def Exec(
        self,
        request: worker_pb2.ExecRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[worker_pb2.ExecChunk]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, bytes, int] | None] = asyncio.Queue()

        def producer() -> None:
            try:
                for kind, payload, exit_code in self._runtime.exec_streaming(
                    container_id=request.container_id,
                    cmd=request.cmd,
                    timeout_sec=request.timeout_sec,
                ):
                    asyncio.run_coroutine_threadsafe(
                        queue.put((kind, payload, exit_code)), loop
                    ).result()
            except Exception as exc:  # pragma: no cover - surfaced through queue
                logger.exception("Exec producer crashed: id=%s", request.container_id[:12])
                asyncio.run_coroutine_threadsafe(
                    queue.put((-1, str(exc).encode("utf-8"), 1)), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        producer_task = loop.run_in_executor(None, producer)

        kind_to_enum = {
            STREAM_STDOUT: worker_pb2.ExecChunk.STDOUT,
            STREAM_STDERR: worker_pb2.ExecChunk.STDERR,
            STREAM_EXIT: worker_pb2.ExecChunk.EXIT,
        }
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, payload, exit_code = item
                if kind == -1:
                    await context.abort(grpc.StatusCode.INTERNAL, payload.decode("utf-8"))
                    return
                yield worker_pb2.ExecChunk(
                    stream=kind_to_enum[kind],
                    data=payload,
                    exit_code=exit_code,
                )
        finally:
            await producer_task

    # ----- file ops -----

    async def WriteFile(
        self,
        request: worker_pb2.WriteFileRequest,
        context: grpc.aio.ServicerContext,
    ) -> worker_pb2.Empty:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._runtime.write_file(
                    container_id=request.container_id,
                    path=request.path,
                    content=request.content,
                    mode=request.mode,
                ),
            )
        except FileNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            raise
        except Exception as exc:
            logger.exception("WriteFile failed: path=%s", request.path)
            await context.abort(grpc.StatusCode.INTERNAL, f"write_file: {exc}")
            raise
        return worker_pb2.Empty()

    async def ReadFile(
        self,
        request: worker_pb2.ReadFileRequest,
        context: grpc.aio.ServicerContext,
    ) -> worker_pb2.FileContent:
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(
                None,
                lambda: self._runtime.read_file(
                    container_id=request.container_id,
                    path=request.path,
                ),
            )
        except FileNotFoundError:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"not found: {request.path}")
            raise
        except IsADirectoryError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"is a directory: {request.path}")
            raise
        except Exception as exc:
            logger.exception("ReadFile failed: path=%s", request.path)
            await context.abort(grpc.StatusCode.INTERNAL, f"read_file: {exc}")
            raise
        return worker_pb2.FileContent(content=content, size=len(content))

    async def ListDir(
        self,
        request: worker_pb2.ListDirRequest,
        context: grpc.aio.ServicerContext,
    ) -> worker_pb2.DirListing:
        loop = asyncio.get_running_loop()
        try:
            entries = await loop.run_in_executor(
                None,
                lambda: self._runtime.list_dir(
                    container_id=request.container_id,
                    path=request.path,
                ),
            )
        except FileNotFoundError:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"not found: {request.path}")
            raise
        except Exception as exc:
            logger.exception("ListDir failed: path=%s", request.path)
            await context.abort(grpc.StatusCode.INTERNAL, f"list_dir: {exc}")
            raise
        proto_entries = [
            worker_pb2.DirEntry(name=e.name, is_dir=e.is_dir, size=e.size)
            for e in entries
        ]
        return worker_pb2.DirListing(entries=proto_entries)

    # ----- health -----

    async def HealthCheck(
        self,
        request: worker_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> worker_pb2.HealthStatus:
        return worker_pb2.HealthStatus(
            healthy=True,
            active_containers=self._runtime.count_active(),
            version=WORKER_VERSION,
        )

    async def ReleaseAll(
        self,
        request: worker_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> worker_pb2.Empty:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._runtime.release_all)
        except Exception as exc:
            logger.exception("ReleaseAll failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"release_all: {exc}")
            raise
        return worker_pb2.Empty()
