import asyncio
import json

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI

from nanorollout.core.config import ServerConfig
from nanorollout.core.models import RunRequest, RunResponse
from nanorollout.core.runners import resolve_request_runner
from nanorollout.core.scheduler import Scheduler, SchedulerTimeoutError


class ResourceManager:
    def __init__(self, concurrency: int) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")
        self._concurrency = asyncio.Semaphore(concurrency)

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[None]:
        await self._concurrency.acquire()
        try:
            yield
        finally:
            self._concurrency.release()


class NanoRolloutServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._resources = ResourceManager(
            concurrency=config.concurrency,
        )
        self._scheduler = Scheduler(
            config.scheduler,
            max_workers=config.concurrency,
        )

    def setup_app(self) -> FastAPI:
        @asynccontextmanager
        async def _lifespan(_: FastAPI):
            self._scheduler._ensure_init()
            yield
            self.close()

        app = FastAPI(
            title="NanoRollout Server",
            description="A Lightweight Framework for Digital Agent Evaluation and Training",
            version="0.1.0",
            lifespan=_lifespan,
        )
        app.post("/run")(self.handle_run)
        app.get("/health")(self.health)
        return app

    def close(self) -> None:
        self._scheduler.close()

    async def health(self) -> dict[str, str]:
        return {"status": "healthy"}

    async def handle_run(self, body: RunRequest) -> RunResponse:
        output_dir = self._build_output_dir(body)

        try:
            async with self._resources.reserve():
                result = await self._scheduler.run(body, output_dir)
        except SchedulerTimeoutError as exc:
            response = self._build_response(body, output_dir, "Timeout", str(exc))
        except Exception as exc:
            response = self._build_response(body, output_dir, "Error", str(exc))
        else:
            response = self._build_success_response(body, result, output_dir)

        await asyncio.to_thread(self._save_result, output_dir, body.instance_id, response)
        return response

    def _build_output_dir(self, body: RunRequest) -> str:
        run_name = (body.run_name or "").strip()
        sample_id = uuid4().hex[:8]
        if not run_name:
            try:
                spec = resolve_request_runner(body)
                runner_path = Path(spec.task) / spec.agent
            except ValueError:
                runner_path = Path(body.task) / (body.runner or body.agent)
            model_name = body.model_name.strip() if body.model_name else "model"
            run_name = str(runner_path / model_name)
        output_dir = Path(
            self.config.output_dir,
            run_name,
            body.instance_id,
            sample_id,
        ).expanduser().resolve()
        return str(output_dir)

    def _build_success_response(
        self, body: RunRequest, result: Any, output_dir: str
    ) -> RunResponse:
        if isinstance(result, RunResponse):
            return result

        if not isinstance(result, dict):
            return self._build_response(body, output_dir, "Completed")

        return RunResponse.model_validate(result)

    def _build_response(
        self,
        body: RunRequest,
        output_dir: str,
        exit_status: str,
        error: str | None = None,
    ) -> RunResponse:
        metadata: dict[str, Any] = {"output_dir": output_dir}
        if error is not None:
            metadata["error"] = error
        if exit_status == "Timeout":
            metadata["timed_out"] = True
            metadata["task_timeout_s"] = body.task_timeout_s
        return RunResponse(
            reward=0.0,
            messages=[],
            exit_status=exit_status,
            metadata=metadata,
        )

    def _save_result(self, output_dir: str, instance_id: str, response: RunResponse) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"{instance_id}.json", "w") as handle:
            json.dump(response.model_dump(), handle, indent=2)
