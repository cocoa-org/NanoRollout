"""Docker runtime adapter for the GCE worker daemon.

Thin sync wrapper around docker-py exposing what WorkerService needs:
lifecycle, streaming exec, file ops, and pool hygiene. Tracks created
containers in _active for release_all cleanup; does NOT track
host-spawned containers."""

from __future__ import annotations

import io
import logging
import shlex
import tarfile
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import docker
from docker.errors import APIError, NotFound

logger = logging.getLogger(__name__)


STREAM_STDOUT = 0
STREAM_STDERR = 1
STREAM_EXIT = 2


@dataclass
class DirEntry:
    name: str
    is_dir: bool
    size: int


class DockerRuntime:
    """At most one active container per worker (data structures accept
    many for future multi-container support)."""

    def __init__(self) -> None:
        self._client = docker.from_env()
        self._active: dict[str, str] = {}  # id -> image (for diagnostics)
        self._lock = threading.Lock()

    # ----- lifecycle -----

    def start_container(
        self,
        image: str,
        work_dir: str,
        env_vars: Optional[dict[str, str]] = None,
        cpu_limit: int = 0,
        memory_limit_mb: int = 0,
    ) -> str:
        """Start a detached container and return its short id.

        ``cpu_limit`` is interpreted as whole CPUs; ``memory_limit_mb`` in
        MiB. Zero means unconstrained on either axis.
        """
        kwargs: dict[str, object] = {
            "command": "/bin/bash",
            "stdin_open": True,
            "tty": True,
            "detach": True,
            "working_dir": work_dir or "/workspace",
            "platform": "linux/amd64",
        }
        if env_vars:
            kwargs["environment"] = dict(env_vars)
        if cpu_limit and cpu_limit > 0:
            kwargs["nano_cpus"] = int(cpu_limit) * 1_000_000_000
        if memory_limit_mb and memory_limit_mb > 0:
            kwargs["mem_limit"] = f"{int(memory_limit_mb)}m"

        container = self._client.containers.run(image, **kwargs)
        time.sleep(0.5)  # let bash settle so subsequent exec_run finds it
        with self._lock:
            self._active[container.id] = image
        logger.info("started container id=%s image=%s", container.id[:12], image)
        return container.id

    def stop_container(self, container_id: str) -> None:
        container = self._lookup(container_id, missing_ok=True)
        if container is None:
            with self._lock:
                self._active.pop(container_id, None)
            return
        try:
            container.stop(timeout=5)
        except APIError as exc:
            logger.warning("stop_container stop error id=%s: %s", container_id[:12], exc)
        try:
            container.remove(force=True)
        except APIError as exc:
            logger.warning("stop_container remove error id=%s: %s", container_id[:12], exc)
        with self._lock:
            self._active.pop(container_id, None)
        logger.info("stopped container id=%s", container_id[:12])

    def release_all(self) -> int:
        with self._lock:
            ids = list(self._active.keys())
        for cid in ids:
            try:
                self.stop_container(cid)
            except Exception as exc:
                logger.warning("release_all error on id=%s: %s", cid[:12], exc)
        return len(ids)

    def count_active(self) -> int:
        with self._lock:
            return len(self._active)

    # ----- exec -----

    def exec_streaming(
        self,
        container_id: str,
        cmd: str,
        timeout_sec: int = 0,
    ) -> Iterator[tuple[int, bytes, int]]:
        """Run ``cmd`` inside the container, yielding chunks as they arrive.

        Yields ``(stream_kind, payload, exit_code)`` where ``stream_kind``
        is one of ``STREAM_STDOUT`` / ``STREAM_STDERR`` / ``STREAM_EXIT``.
        For STDOUT/STDERR ``exit_code`` is 0 (placeholder); for EXIT
        ``payload`` is empty and ``exit_code`` is the real exit status.

        ``timeout_sec`` <= 0 means no GNU-coreutils ``timeout`` wrapping;
        the caller can still impose a deadline on the iterator.
        """
        container = self._lookup(container_id)
        api = self._client.api

        if timeout_sec and timeout_sec > 0:
            wrapped = ["timeout", f"{int(timeout_sec)}s", "/bin/bash", "-lc", cmd]
        else:
            wrapped = ["/bin/bash", "-lc", cmd]

        exec_dict = api.exec_create(container.id, cmd=wrapped, tty=False, stdout=True, stderr=True)
        exec_id = exec_dict["Id"]

        # NOTE (B.6): demux=False mirrors docker.py to ensure stdout/stderr
        # arrive in chronological order, preserving extract_cwd_marker's
        # "marker at end of last line" assumption. With demux=True the two
        # streams arrived separately and concatenation order made the cwd
        # marker appear in the middle of the combined output, swallowing
        # any post-marker stderr (e.g. python tracebacks). STREAM_STDERR
        # stays in the proto for future demux'd-stream callers.
        stream = api.exec_start(exec_id, stream=True, demux=False)
        try:
            for chunk in stream:
                if chunk:
                    yield STREAM_STDOUT, chunk, 0
        finally:
            try:
                stream.close()  # type: ignore[attr-defined]
            except Exception:
                pass

        info = api.exec_inspect(exec_id)
        exit_code = int(info.get("ExitCode") or 0)
        yield STREAM_EXIT, b"", exit_code

    # ----- file ops -----

    def write_file(
        self,
        container_id: str,
        path: str,
        content: bytes,
        mode: int = 0,
    ) -> None:
        container = self._lookup(container_id)
        if not path.startswith("/"):
            raise ValueError(f"path must be absolute: {path!r}")

        parent = path.rsplit("/", 1)[0] or "/"
        # Ensure parent exists. Cheap to call mkdir -p even if it does.
        api = self._client.api
        mkdir_exec = api.exec_create(
            container.id,
            cmd=["/bin/sh", "-c", f"mkdir -p {shlex.quote(parent)}"],
            stdout=True,
            stderr=True,
        )
        api.exec_start(mkdir_exec["Id"], stream=False)

        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.lstrip("/").rsplit("/", 1)[-1])
            info.size = len(content)
            info.mtime = int(time.time())
            info.mode = mode if mode else 0o644
            tar.addfile(info, io.BytesIO(content))
        tar_buf.seek(0)

        ok = container.put_archive(parent, tar_buf.getvalue())
        if not ok:
            raise RuntimeError(f"put_archive returned False for {path!r}")

    def read_file(self, container_id: str, path: str) -> bytes:
        container = self._lookup(container_id)
        try:
            bits, stat = container.get_archive(path)
        except NotFound:
            raise FileNotFoundError(path)
        buf = io.BytesIO()
        for chunk in bits:
            buf.write(chunk)
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r") as tar:
            members = tar.getmembers()
            if not members:
                raise FileNotFoundError(path)
            member = members[0]
            if member.isdir():
                raise IsADirectoryError(path)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(path)
            return extracted.read()

    def list_dir(self, container_id: str, path: str) -> list[DirEntry]:
        container = self._lookup(container_id)
        # Use find with -printf to get one entry per line in a stable format.
        # -mindepth 1 -maxdepth 1 = direct children only.
        api = self._client.api
        cmd = [
            "/bin/sh",
            "-c",
            f"find {shlex.quote(path)} -mindepth 1 -maxdepth 1 -printf '%f\\t%y\\t%s\\n'",
        ]
        exec_dict = api.exec_create(container.id, cmd=cmd, stdout=True, stderr=True)
        out = api.exec_start(exec_dict["Id"], stream=False)
        info = api.exec_inspect(exec_dict["Id"])
        if info.get("ExitCode"):
            raise FileNotFoundError(path)
        text = (out or b"").decode("utf-8", errors="replace")

        entries: list[DirEntry] = []
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            name, kind, size_s = parts
            try:
                size_i = int(size_s)
            except ValueError:
                size_i = 0
            entries.append(DirEntry(name=name, is_dir=(kind == "d"), size=0 if kind == "d" else size_i))
        entries.sort(key=lambda e: e.name)
        return entries

    # ----- helpers -----

    def _lookup(self, container_id: str, missing_ok: bool = False):
        try:
            return self._client.containers.get(container_id)
        except NotFound:
            if missing_ok:
                return None
            raise
