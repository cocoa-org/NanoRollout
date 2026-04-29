import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from tinyflow.api.models import RunRequest, RunResponse
from tinyflow.core.config import ServerConfig
from tinyflow.runtime.runners import RunnerSpec

from tinyflow.runtime.backends.ray_scheduler import RayScheduler, SchedulerTimeoutError

logger = logging.getLogger(__name__)


class TinyFlowServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.sem = asyncio.Semaphore(config.concurrency)
        self._app: Optional[FastAPI] = None

        # TODO: add other backends here
        self._scheduler = RayScheduler(
            config.ray,
            max_workers=config.concurrency,
            monitoring=config.monitoring,
        )
        self._pending_lock = asyncio.Lock()
        self._pending = 0
        self._max_pending_jobs = config.max_pending_jobs
        self._cache_id = uuid4().hex

    def setup_app(self) -> FastAPI:
        @asynccontextmanager
        async def _lifespan(_: FastAPI):
            try:
                self._scheduler.warmup()
                logger.info("TinyFlow scheduler warmup completed.")
            except Exception:
                logger.exception("TinyFlow scheduler warmup failed.")
            try:
                yield
            finally:
                try:
                    self._scheduler.close()
                    logger.info("TinyFlow scheduler closed.")
                except Exception:
                    logger.exception("TinyFlow scheduler shutdown failed.")

        app = FastAPI(
            title="TinyFlowServer",
            description="Lightweight agent server for Miles training",
            version="0.1.0",
            lifespan=_lifespan,
        )

        app.post("/run")(self.handle_run)
        app.get("/health")(self.health)

        self._app = app
        return app

    async def health(self) -> Dict[str, str]:
        return {"status": "healthy"}

    async def handle_run(self, body: RunRequest) -> RunResponse:
        if self._max_pending_jobs > 0:
            acquired = await self._try_acquire_pending()
            if not acquired:
                raise HTTPException(status_code=429, detail="Too many pending jobs.")
        try:
            async with self.sem:
                return await self._execute_agent(body)
        finally:
            if self._max_pending_jobs > 0:
                await self._release_pending()

    async def _try_acquire_pending(self) -> bool:
        async with self._pending_lock:
            if self._pending >= self._max_pending_jobs:
                return False
            self._pending += 1
            return True

    async def _release_pending(self) -> None:
        async with self._pending_lock:
            self._pending = max(0, self._pending - 1)

    async def _execute_agent(self, body: RunRequest) -> RunResponse:
        instance_id = body.instance_id
        self._setup_env_vars()
        run_name = (body.run_name or "").strip()
        sample_id = uuid4().hex[:8]
        if not run_name:
            runner = (body.runner or "default").strip() or "default"
            model_name = body.model_name.strip() if body.model_name else "model"
            run_name = str(Path(runner) / model_name)
        output_dir = Path(self.config.output_dir, run_name, instance_id, sample_id).expanduser().resolve()
        output_dir = str(output_dir)

        runner_spec = self._build_runner_spec(body)
        params = self._build_runner_params(body, output_dir)
        runtime_env = self._resolve_runtime_env(body)

        try:
            start_time = time.time()
            result = await self._scheduler.run(
                runner_spec,
                params,
                output_dir,
                resources=body.resources,
                runtime_env=runtime_env,
                task_timeout_s=body.task_timeout_s,
            )
            _ = time.time() - start_time

        except SchedulerTimeoutError as exc:
            logger.warning("Task timed out for %s: %s", instance_id, exc)
            return self._build_timeout_response(body, str(exc), output_dir)
        except Exception as exc:
            logger.exception("Error executing agent for %s", instance_id)
            return self._build_error_response(body, str(exc), output_dir)

        if isinstance(result, RunResponse):
            response = result
        else:
            response = RunResponse(**result)

        await asyncio.to_thread(self._save_result, output_dir, instance_id, response)
        return response

    def _build_runner_spec(self, body: RunRequest) -> RunnerSpec:
        return RunnerSpec(
            runner_type=body.runner,
            runner_module=body.runner_module,
            runner_entrypoint=body.runner_entrypoint,
        )

    def _resolve_runtime_env(self, body: RunRequest) -> Optional[Dict[str, Any]]:
        base_env = dict(self.config.ray.runtime_env or {})
        override_env = body.runtime_env if isinstance(body.runtime_env, dict) else {}
        runtime_env = self._merge_runtime_env(base_env, override_env)
        return runtime_env or None

    @staticmethod
    def _merge_runtime_env(
        base_env: Dict[str, Any], override_env: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not base_env and not override_env:
            return {}
        merged: Dict[str, Any] = dict(base_env)
        for key, value in override_env.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    def _build_runner_params(self, body: RunRequest, output_dir: str) -> Dict[str, Any]:
        params = {
            "instance_id": body.instance_id,
            "model_name": body.model_name,
            "base_url": body.base_url,
            "api_key": body.api_key,
            "env_type": body.env_type,
            "sampling_params": json.dumps(body.sampling_params or {}),
            "output_dir": output_dir,
            "extra_args": body.extra_args,
        }
        return params

    def _build_timeout_response(
        self, body: RunRequest, error: str, output_dir: str
    ) -> RunResponse:
        response = RunResponse(
            reward=0.0,
            messages=[],
            exit_status="Timeout",
            metadata={
                "error": error,
                "timed_out": True,
                "task_timeout_s": body.task_timeout_s,
            },
        )
        self._save_result(output_dir, body.instance_id, response)
        return response

    def _build_error_response(
        self, body: RunRequest, error: str, output_dir: str
    ) -> RunResponse:
        response = RunResponse(
            reward=0.0,
            messages=[],
            exit_status="Error",
            metadata={"error": error},
        )
        self._save_result(output_dir, body.instance_id, response)
        return response

    def _save_result(self, output_dir: str, instance_id: str, response: RunResponse) -> None:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f"{instance_id}.json", "w") as handle:
                json.dump(response.model_dump(), handle, indent=2)

        except Exception as exc:
            logger.warning("Failed to save result for %s: %s", instance_id, exc)

    def _setup_env_vars(self) -> None:
        pass  # TODO: add env vars here
