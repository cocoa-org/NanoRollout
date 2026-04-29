import gc
import os
import ray
import asyncio
import threading
import traceback
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from ray.exceptions import GetTimeoutError, TaskCancelledError

from tinyflow.api.models import ResourceRequest
from tinyflow.runtime.runners import RunnerRegistry, RunnerSpec
from tinyflow.core.config import MonitoringConfig, RayRuntimeConfig
from tinyflow.runtime.monitoring import ClusterResourceMonitor


class SchedulerTimeoutError(TimeoutError):
    pass


class RayScheduler:

    def __init__(
        self,
        config: RayRuntimeConfig,
        max_workers: Optional[int] = None,
        monitoring: Optional[MonitoringConfig] = None,
    ) -> None:
        self.config = config
        self._initialized = False
        self._init_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        max_workers = max_workers or os.cpu_count()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._monitor = self._build_monitor(monitoring)
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
            if self._monitor:
                self._monitor.start()

    def warmup(self) -> None:
        """Eagerly initialize Ray and monitoring resources."""
        self._ensure_init()

    def _build_monitor(
        self, monitoring: Optional[MonitoringConfig]
    ) -> Optional[ClusterResourceMonitor]:
        if monitoring is None or not monitoring.enabled:
            return None
        wandb_cfg = monitoring.wandb
        return ClusterResourceMonitor(
            enabled=monitoring.enabled,
            interval_s=monitoring.interval_s,
            wandb_enabled=wandb_cfg.enabled,
            wandb_project=wandb_cfg.project,
            wandb_entity=wandb_cfg.entity,
            wandb_run_name=wandb_cfg.run_name,
            wandb_group=wandb_cfg.group,
            wandb_job_type=wandb_cfg.job_type,
            wandb_tags=list(wandb_cfg.tags),
            wandb_config=dict(wandb_cfg.config),
            print_summary=monitoring.print_summary,
        )

    async def run(
        self,
        runner_spec: RunnerSpec,
        params: Dict[str, Any],
        output_dir: str,
        resources: Optional[ResourceRequest] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
        task_timeout_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_blocking,
            runner_spec,
            params,
            output_dir,
            resources,
            runtime_env,
            task_timeout_s,
        )

    def _run_blocking(
        self,
        runner_spec: RunnerSpec,
        params: Dict[str, Any],
        output_dir: str,
        resources: Optional[ResourceRequest],
        runtime_env: Optional[Dict[str, Any]],
        task_timeout_s: Optional[int],
    ) -> Dict[str, Any]:
        self._ensure_init()
        payload = {
            "runner_spec": runner_spec.model_dump(),
            "params": params,
            "output_dir": output_dir,
            "num_cpus": resources.num_cpus if resources else None,
        }
        opts = self._build_options(resources)
        if runtime_env:
            opts = dict(opts)
            opts["runtime_env"] = runtime_env
        print(f"opts: {opts}")
        worker = _ray_worker.options(**opts) if opts else _ray_worker
        result_ref = worker.remote(payload)
        if task_timeout_s is not None and task_timeout_s > 0:
            try:
                result = ray.get(result_ref, timeout=float(task_timeout_s))
            except GetTimeoutError as exc:
                ray.cancel(result_ref, force=False, recursive=True)
                # Give the task a short grace period so runner-level finally blocks can run.
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
        if self._monitor:
            self._monitor.stop()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _build_options(self, resources: Optional[ResourceRequest]) -> Dict[str, Any]:
        if not resources:
            return {}
        opts: Dict[str, Any] = {}
        for key in ("num_cpus", "num_gpus"):
            value = getattr(resources, key)
            if value is not None:
                opts[key] = value
        if resources.resources:
            opts["resources"] = resources.resources
        memory_gb = resources.memory_gb
        if memory_gb is not None and memory_gb > 0:
            opts["memory"] = int(memory_gb * 1024**3)
        return opts


def _apply_cpu_affinity(num_cpus: Optional[int]) -> None:
    """Restrict this process to use only `num_cpus` CPU cores.
    
    Uses process ID to distribute workers across different CPU ranges,
    avoiding all workers competing for the same cores.
    """
    if num_cpus is None or num_cpus <= 0:
        return
    try:
        available_cpus = sorted(os.sched_getaffinity(0))
        total_cpus = len(available_cpus)
        if total_cpus <= num_cpus:
            return  # Already limited enough
        
        # Use PID to calculate offset, distribute workers across CPU ranges
        pid = os.getpid()
        num_slots = total_cpus // num_cpus  # How many worker slots we can have
        slot_id = pid % num_slots  # Which slot this worker gets
        start_idx = slot_id * num_cpus
        
        limited_cpus = set(available_cpus[start_idx:start_idx + num_cpus])
        os.sched_setaffinity(0, limited_cpus)
        print(f"[CPU Affinity] PID={pid}, slot={slot_id}, CPUs: {sorted(limited_cpus)}")
    except (OSError, AttributeError) as e:
        print(f"[CPU Affinity] Failed to set affinity: {e}")


_MAX_CALLS_PER_WORKER = 50


def _build_worker():

    @ray.remote(max_calls=_MAX_CALLS_PER_WORKER)
    def _ray_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            runner_spec = RunnerSpec(**payload["runner_spec"])
            params = payload["params"]
            runner = RunnerRegistry.create(runner_spec)
            result = runner.run(params)
            del runner, params, runner_spec, payload
            gc.collect()
            return {"result": result}
        except Exception as exc:
            return {"error": str(exc), "traceback": traceback.format_exc()}

    return _ray_worker


_ray_worker = _build_worker()
