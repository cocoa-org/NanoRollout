import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import uuid4

from brew.core.models import RunRequest, RunResponse
from brew.core.runners import (
    build_runner_params,
    load_runner_callable,
    resolve_request_runner,
)

logger = logging.getLogger(__name__)


def build_output_dir(output_root: str, request: RunRequest) -> str:
    run_name = (request.run_name or "").strip()
    sample_id = uuid4().hex[:8]
    if not run_name:
        spec = resolve_request_runner(request)
        model_name = request.model_name.strip() if request.model_name else "model"
        run_name = str(Path(spec.task) / spec.agent / model_name)
    return str(
        Path(output_root, run_name, request.instance_id, sample_id)
        .expanduser()
        .resolve()
    )


def save_response(output_dir: str, response: RunResponse) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{response.instance_id}.json", "w") as handle:
        json.dump(response.model_dump(), handle, indent=2)


def build_response(
    request: RunRequest,
    output_dir: str,
    exit_status: str,
    error: str | None = None,
) -> RunResponse:
    return RunResponse(
        instance_id=request.instance_id,
        exit_status=exit_status,
        output_dir=output_dir,
        error=error,
    )


def build_success_response(
    request: RunRequest,
    result: Any,
    output_dir: str,
) -> RunResponse:
    if isinstance(result, RunResponse):
        response = result.model_copy()
        response.instance_id = request.instance_id
        response.output_dir = response.output_dir or output_dir
        return response

    if not isinstance(result, dict):
        return build_response(request, output_dir, "Completed")

    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    return build_response(
        request=request,
        output_dir=output_dir,
        exit_status=str(result.get("exit_status") or "Completed"),
        error=result.get("error") or metadata.get("error"),
    )


def _run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        runner = load_runner_callable(payload["module_name"], payload["entrypoint"])
        result = runner(**payload["params"])
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


@contextmanager
def _suppress_child_console() -> Iterator[None]:
    """Keep noisy task process output out of the parent console."""
    # TODO(Junli): Handle subprocess logging
    previous_stdout = sys.stdout
    previous_stderr = sys.stderr
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level

    with open(os.devnull, "w", encoding="utf-8") as handle:
        sys.stdout = handle
        sys.stderr = handle
        root.handlers = []
        try:
            yield
        finally:
            logging.shutdown()
            root.handlers = previous_handlers
            root.setLevel(previous_level)
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr


def _process_entry(payload: dict[str, Any], result_queue) -> None:
    with _suppress_child_console():
        result_queue.put(_run_payload(payload))


def _truncate(value: str, max_length: int = 44) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


@dataclass
class _ActiveRun:
    request: RunRequest
    output_dir: str
    process: Any
    result_queue: Any
    started: float
    status: str = "running"


class LocalProcessRunner:
    def __init__(
        self,
        output_root: str,
        concurrency: int = 1,
        poll_interval: float = 0.2,
        show_progress: bool = False,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")
        self.output_root = output_root
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.show_progress = show_progress
        self._context = mp.get_context()

    def run_many(self, requests: Iterable[RunRequest]) -> list[RunResponse]:
        pending = list(requests)
        if self.show_progress:
            return self._run_many_with_progress(pending)
        return self._run_many(pending)

    def _run_many(self, pending: list[RunRequest]) -> list[RunResponse]:
        active: list[_ActiveRun] = []
        responses: list[RunResponse] = []

        try:
            while pending or active:
                while pending and len(active) < self.concurrency:
                    request = pending.pop(0)
                    active.append(self._start(request))

                made_progress = False
                for run in list(active):
                    response = self._poll(run)
                    if response is None:
                        continue
                    active.remove(run)
                    save_response(run.output_dir, response)
                    responses.append(response)
                    made_progress = True

                if active and not made_progress:
                    time.sleep(self.poll_interval)
        finally:
            for run in active:
                if run.process.is_alive():
                    run.process.terminate()
                run.process.join()

        return responses

    def _run_many_with_progress(self, pending: list[RunRequest]) -> list[RunResponse]:
        from rich.console import Group
        from rich.live import Live
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
        from rich.table import Table

        total = len(pending)
        status_counts = {"completed": 0, "failed": 0, "timeout": 0}
        progress = Progress(
            TextColumn("[bold]tbrew run[/bold]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
        )
        task_id = progress.add_task("tasks", total=total)

        def render(active: list[_ActiveRun]) -> Group:
            table = Table(title="Active Jobs", expand=True)
            table.add_column("Instance")
            table.add_column("PID", justify="right")
            table.add_column("Elapsed", justify="right")
            table.add_column("Status")
            table.add_column("Output")
            for run in active:
                elapsed = int(time.time() - run.started)
                table.add_row(
                    _truncate(run.request.instance_id),
                    str(run.process.pid or "-"),
                    f"{elapsed}s",
                    run.status,
                    str(Path(run.output_dir).relative_to(Path(self.output_root).resolve())),
                )
            if not active:
                table.add_row("-", "-", "-", "idle", "-")

            summary = Table.grid(expand=True)
            summary.add_column()
            summary.add_row(
                f"completed={status_counts['completed']} "
                f"failed={status_counts['failed']} "
                f"timeout={status_counts['timeout']} "
                f"pending={len(pending)}"
            )
            return Group(progress, summary, table)

        with Live(render([]), refresh_per_second=4) as live:
            active: list[_ActiveRun] = []
            responses: list[RunResponse] = []
            try:
                while pending or active:
                    while pending and len(active) < self.concurrency:
                        request = pending.pop(0)
                        active.append(self._start(request))

                    made_progress = False
                    for run in list(active):
                        response = self._poll(run)
                        if response is None:
                            continue
                        active.remove(run)
                        save_response(run.output_dir, response)
                        responses.append(response)
                        progress.advance(task_id)
                        if response.exit_status.lower() == "timeout":
                            status_counts["timeout"] += 1
                        elif response.error or response.exit_status.lower() == "error":
                            status_counts["failed"] += 1
                        else:
                            status_counts["completed"] += 1
                        made_progress = True

                    live.update(render(active))
                    if active and not made_progress:
                        time.sleep(self.poll_interval)
            finally:
                for run in active:
                    if run.process.is_alive():
                        run.process.terminate()
                    run.process.join()

        return responses

    def _start(self, request: RunRequest) -> _ActiveRun:
        spec = resolve_request_runner(request)
        output_dir = build_output_dir(self.output_root, request)
        payload = {
            "module_name": spec.module,
            "entrypoint": spec.entrypoint,
            "params": build_runner_params(request, output_dir),
        }
        result_queue = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_process_entry,
            args=(payload, result_queue),
        )
        process.start()
        if not self.show_progress:
            logger.info(
                "[%s] started pid=%s runner=%s output_dir=%s",
                request.instance_id,
                process.pid,
                f"{spec.task}/{spec.agent}",
                output_dir,
            )
        return _ActiveRun(
            request=request,
            output_dir=output_dir,
            process=process,
            result_queue=result_queue,
            started=time.time(),
        )

    def _poll(self, run: _ActiveRun) -> RunResponse | None:
        timeout_s = run.request.task_timeout_s
        if timeout_s and time.time() - run.started > timeout_s:
            self._terminate(run)
            return build_response(
                run.request,
                run.output_dir,
                "Timeout",
                f"Task exceeded timeout: {timeout_s}s",
            )

        try:
            payload = run.result_queue.get_nowait()
        except queue.Empty:
            if run.process.is_alive():
                return None
            run.process.join()
            try:
                payload = run.result_queue.get(timeout=0.1)
            except queue.Empty:
                return build_response(
                    run.request,
                    run.output_dir,
                    "Error",
                    f"Process exited with code {run.process.exitcode} without a result.",
                )
        else:
            run.process.join()

        if isinstance(payload, dict) and payload.get("error"):
            error = payload.get("error")
            if payload.get("traceback"):
                error = f"{error}\n{payload['traceback']}"
            return build_response(
                run.request,
                run.output_dir,
                "Error",
                str(error),
            )
        return build_success_response(
            run.request,
            payload.get("result") if isinstance(payload, dict) else payload,
            run.output_dir,
        )

    def _terminate(self, run: _ActiveRun) -> None:
        if run.process.is_alive():
            run.process.terminate()
            run.process.join(timeout=5)
        if run.process.is_alive():
            run.process.kill()
            run.process.join()
