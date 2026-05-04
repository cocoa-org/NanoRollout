import atexit
import asyncio
import gc
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from brew.core.config import SchedulerConfig
from brew.core.models import ResourceRequest, RunRequest
from brew.core.runners import (
    build_runner_params,
    load_runner_callable,
    resolve_request_runner,
)

import ray
from ray.exceptions import GetTimeoutError, TaskCancelledError


class SchedulerTimeoutError(TimeoutError):
    pass


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        max_workers: int | None = None,
    ) -> None:
        self.config = config
        self._initialized = False
        self._init_lock = threading.Lock()
        pool_size = max_workers or os.cpu_count() or 1
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=pool_size
        )
        atexit.register(self.close)

    def _ensure_init(self) -> None:
        if self._initialized and ray.is_initialized():
            return
        with self._init_lock:
            if self._initialized and ray.is_initialized():
                return
            ray.init(
                address=self.config.address,
                namespace=self.config.namespace,
                runtime_env=self.config.runtime_env,
            )
            self._initialized = True

    async def run(
        self,
        request: RunRequest,
        output_dir: str,
    ) -> dict[str, Any]:
        spec = resolve_request_runner(request)
        params = build_runner_params(request, output_dir)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_blocking,
            spec.module,
            spec.entrypoint,
            params,
            request.resources,
            request.task_timeout_s,
        )

    def _run_blocking(
        self,
        module_name: str,
        entrypoint: str,
        params: dict[str, Any],
        resources: ResourceRequest | None,
        task_timeout_s: int | None,
    ) -> dict[str, Any]:
        self._ensure_init()
        payload = {
            "module_name": module_name,
            "entrypoint": entrypoint,
            "params": params,
        }
        opts = self._build_options(resources)

        worker = _ray_worker.options(**opts) if opts else _ray_worker
        result_ref = worker.remote(payload)
        if task_timeout_s is not None and task_timeout_s > 0:
            try:
                result = ray.get(result_ref, timeout=float(task_timeout_s))
            except GetTimeoutError as exc:
                ray.cancel(result_ref, force=False, recursive=True)
                try:
                    ray.get(result_ref, timeout=10)
                except TaskCancelledError:
                    pass
                except GetTimeoutError:
                    ray.cancel(result_ref, force=True, recursive=True)
                except Exception:
                    ray.cancel(result_ref, force=True, recursive=True)
                raise SchedulerTimeoutError(
                    f"Task exceeded timeout: {task_timeout_s}s"
                ) from exc
        else:
            result = ray.get(result_ref)

        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"Ray task failed: {result.get('error')}\n{result.get('traceback', '')}"
            )
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result if isinstance(result, dict) else {}

    def close(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self._initialized and ray.is_initialized():
            ray.shutdown()
            self._initialized = False

    def _build_options(self, resources: ResourceRequest | None) -> dict[str, Any]:
        if not resources:
            return {}
        opts: dict[str, Any] = {}
        for key in ("num_cpus", "num_gpus"):
            value = getattr(resources, key)
            if value is not None:
                opts[key] = value
        if resources.custom:
            opts["resources"] = resources.custom
        memory_gb = resources.memory_gb
        if memory_gb is not None and memory_gb > 0:
            opts["memory"] = int(memory_gb * 1024**3)
        return opts

_MAX_CALLS_PER_WORKER = 50


def _build_worker():
    @ray.remote(max_calls=_MAX_CALLS_PER_WORKER)
    def _ray_worker(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            runner = load_runner_callable(
                payload["module_name"],
                payload["entrypoint"],
            )
            params = payload["params"]
            result = runner(**params)
            del runner, params, payload
            gc.collect()
            return {"result": result}
        except Exception as exc:
            return {"error": str(exc), "traceback": traceback.format_exc()}

    return _ray_worker


_ray_worker = _build_worker()
