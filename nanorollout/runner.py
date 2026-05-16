"""Common task runner lifecycle."""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskRunRequest:
    instance_id: str
    output_dir: str
    model_name: str
    base_url: Optional[str]
    api_key: Optional[str]
    env_type: str
    sampling_params: Optional[object]
    extra_args: Dict[str, Any]


@dataclass
class TaskSpec:
    id: str
    kind: str
    instruction: str
    payload: Any = None
    environment: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskAdapter(ABC):
    """Adapter contract for the common prepare/env/agent/eval/result flow."""

    runner_label: str
    env_logger_name: Optional[str] = None
    eval_logger_name: Optional[str] = None

    @abstractmethod
    def create_environment(
        self,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> Any:
        """Create and return the task environment."""

    def start_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        del task
        set_tool_log_context = getattr(env_obj, "set_tool_log_context", None)
        if callable(set_tool_log_context):
            set_tool_log_context(request.instance_id)

        start = getattr(env_obj, "start", None)
        if callable(start):
            start()

    def stop_environment(
        self,
        env_obj: Any,
        task: Optional[TaskSpec],
        request: TaskRunRequest,
    ) -> None:
        del task, request
        for method_name in ("stop", "close", "cleanup_environment"):
            method = getattr(env_obj, method_name, None)
            if callable(method):
                method()
                return

    def describe_environment(
        self,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> str:
        del task
        return request.env_type

    @abstractmethod
    def prepare_task(
        self,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> TaskSpec:
        """Resolve benchmark task data before creating the environment."""

    def setup_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        """Run benchmark-specific environment setup before agent execution."""

    @abstractmethod
    def build_agent(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> Any:
        """Construct the agent for a normalized task spec."""

    @abstractmethod
    def run_agent(
        self,
        agent: Any,
        task: TaskSpec,
        env_obj: Any,
    ) -> Any:
        """Run the agent and return its raw result."""

    def after_agent_result(self, agent_result: Any) -> None:
        """Normalize or enrich an agent result before evaluation."""

    @abstractmethod
    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Run benchmark evaluation and return payload plus raw output."""

    @abstractmethod
    def build_reward_payload(
        self,
        instance_id: str,
        eval_payload: Dict[str, Any],
        error_msg: Optional[str],
    ) -> Dict[str, Any]:
        """Convert eval output into the public reward payload."""

    def write_result(
        self,
        trial_dir: Path,
        instance_id: str,
        model: str,
        base_url: Optional[str],
        env_type: str,
        agent_result: Any,
        tools_json: Optional[Dict[str, Any]],
        reward_payload: Dict[str, Any],
        eval_output: Optional[str],
        started: float,
        metadata: Dict[str, Any],
    ) -> None:
        write_task_artifacts(
            trial_dir,
            instance_id,
            model,
            base_url,
            env_type,
            agent_result,
            tools_json,
            reward_payload,
            eval_output,
            started,
            metadata,
        )

    def get_tools_json(self, agent: Any) -> Optional[Dict[str, Any]]:
        return get_agent_tools_json(agent)

    def build_agent_metrics(
        self,
        messages: list[Dict[str, Any]],
        agent_time: float,
        eval_time: float,
        total_time: float,
        agent_result: Any,
        eval_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        del agent_result, eval_payload
        return build_agent_metrics(messages, agent_time, eval_time, total_time)

    def update_metadata(
        self,
        metadata: Dict[str, Any],
        task: Optional[TaskSpec],
        agent_result: Any,
        eval_payload: Dict[str, Any],
        error_msg: Optional[str],
    ) -> Dict[str, Any]:
        del task, agent_result, eval_payload, error_msg
        return metadata

    @abstractmethod
    def build_exit_status(
        self,
        error_msg: Optional[str],
        agent_result: Any,
        eval_payload: Dict[str, Any],
    ) -> str:
        """Return the public exit status for the run."""


def require_args(extra_args: Dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if key not in extra_args:
            raise ValueError(f"Missing required {label} argument: {key}")


def count_tool_calls(messages: list[Dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            count += len(tool_calls)
    return count


def build_agent_metrics(
    messages: list[Dict[str, Any]],
    agent_time: float,
    eval_time: float,
    total_time: float,
) -> Dict[str, Any]:
    turns = sum(1 for msg in messages if msg.get("role") == "assistant")
    tool_calls = count_tool_calls(messages)
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "model_query_time_sum": 0.0,
        "env_execution_time_sum": 0.0,
        "eval_time": eval_time,
        "agent_run_time": agent_time,
        "total_time": total_time,
    }


def get_agent_tools_json(agent: Any) -> Optional[Dict[str, Any]]:
    get_tools_schema = getattr(agent, "get_tools_schema", None)
    if not callable(get_tools_schema):
        return None
    tools_schema = get_tools_schema()
    return tools_schema if tools_schema else None


class ThreadLogFilter(logging.Filter):
    """Filter log records to one runner thread."""

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self._thread_id


@contextmanager
def trial_logging(trial_dir: Path) -> Iterator[Path]:
    log_path = trial_dir / "trial.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handler.addFilter(ThreadLogFilter(threading.get_ident()))

    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield log_path
    finally:
        root.removeHandler(handler)
        handler.close()


@contextmanager
def env_logging(
    run_dir: Path,
    logger_name: Optional[str],
    filename: str = "env.log",
) -> Iterator[Path]:
    log_path = run_dir / filename
    if not logger_name:
        yield log_path
        return

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    env_logger = logging.getLogger(logger_name)
    prev_level = env_logger.level
    prev_propagate = env_logger.propagate
    env_logger.setLevel(logging.INFO)
    env_logger.propagate = False
    env_logger.addHandler(handler)
    try:
        yield log_path
    finally:
        env_logger.removeHandler(handler)
        env_logger.setLevel(prev_level)
        env_logger.propagate = prev_propagate
        handler.close()


@contextmanager
def eval_logging(trial_dir: Path, logger_name: Optional[str]) -> Iterator[Path]:
    log_path = trial_dir / "eval.log"
    if not logger_name:
        yield log_path
        return

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    eval_logger = logging.getLogger(logger_name)
    prev_level = eval_logger.level
    eval_logger.setLevel(logging.INFO)
    eval_logger.addHandler(handler)
    try:
        yield log_path
    finally:
        eval_logger.removeHandler(handler)
        eval_logger.setLevel(prev_level)
        handler.close()


def ensure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        root.setLevel(logging.INFO)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_trajectory(trial_dir: Path, payload: dict[str, Any]) -> Path:
    path = trial_dir / "trajectory.json"
    write_json(path, payload)
    return path


def build_reward_payload(
    instance_id: str,
    eval_payload: Dict[str, Any],
    error_msg: Optional[str],
    *,
    default_status: str,
) -> Dict[str, Any]:
    return {
        "instance_id": instance_id,
        "resolved": eval_payload.get("resolved", False),
        "resolved_status": eval_payload.get("resolved_status", default_status),
        "reward": eval_payload.get("reward", 0),
        "eval_exit_code": eval_payload.get("eval_exit_code"),
        "error": eval_payload.get("error") or error_msg,
        "status_map": eval_payload.get("status_map", {}),
        "report": eval_payload.get("report", {}),
    }


def build_metadata(
    instance_id: str,
    env_type: str,
    eval_payload: Dict[str, Any],
    error_msg: Optional[str],
    trial_dir: Optional[Path],
    eval_output: Optional[str],
    agent_result: Any,
    reward_payload: Dict[str, Any],
    *,
    setup_error_status: str = "setup_error",
) -> Dict[str, Any]:
    exit_status = getattr(agent_result, "exit_status", None) if agent_result else None
    metadata = {
        "instance_id": instance_id,
        "environment": env_type,
        "resolved": eval_payload.get("resolved"),
        "resolved_status": eval_payload.get("resolved_status"),
        "eval_exit_code": eval_payload.get("eval_exit_code"),
        "error": error_msg or eval_payload.get("error"),
        "trial_dir": str(trial_dir) if trial_dir else None,
        "has_eval_output": eval_output is not None,
        "reward_payload": reward_payload,
        "exit_status": exit_status
        or (setup_error_status if agent_result is None else None),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def write_task_artifacts(
    trial_dir: Path,
    instance_id: str,
    model: str,
    base_url: Optional[str],
    env_type: str,
    agent_result: Any,
    tools_json: Optional[Dict[str, Any]],
    reward_payload: Dict[str, Any],
    eval_output: Optional[str],
    started: float,
    metadata: Dict[str, Any],
) -> None:
    exit_status = metadata.get("exit_status")
    reward_payload_to_write = dict(reward_payload)
    if exit_status is not None:
        reward_payload_to_write["exit_status"] = exit_status

    traj_payload = {
        "instance_id": instance_id,
        "model": model,
        "api_base": base_url,
        "environment": env_type,
        "success": bool(_agent_value(agent_result, "success", False)),
        "message": _agent_value(agent_result, "message", ""),
        "iterations": _agent_value(agent_result, "iterations", 0),
        "error": _agent_value(agent_result, "error", None) or metadata.get("error"),
        "messages": _agent_messages(agent_result),
        "llm_metrics": _agent_value(agent_result, "llm_metrics", []),
        "llm_cost_total": _agent_value(agent_result, "llm_cost_total", 0.0),
        "wall_time_sec": round(time.time() - started, 2),
        "tools": tools_json,
    }
    if exit_status is not None:
        traj_payload["exit_status"] = exit_status

    write_trajectory(trial_dir, traj_payload)
    write_json(trial_dir / "reward.json", reward_payload_to_write)

    if eval_output is not None:
        (trial_dir / "eval_output.txt").write_text(eval_output, encoding="utf-8")


def resolve_completed_status(eval_payload: Dict[str, Any]) -> str:
    if eval_payload.get("resolved"):
        return "Resolved"
    return "Completed"


def _agent_value(agent_result: Any, field_name: str, default: Any) -> Any:
    if agent_result is None:
        return default
    return getattr(agent_result, field_name, default)


def _agent_messages(agent_result: Any) -> list[Dict[str, Any]]:
    return list(_agent_value(agent_result, "history", []) or [])


def _stop_environment(
    adapter: TaskAdapter,
    env_obj: Any,
    task: Optional[TaskSpec],
    request: TaskRunRequest,
) -> None:
    try:
        adapter.stop_environment(env_obj, task, request)
    except Exception:
        logger.exception(
            "%s cleanup failed for %s", adapter.runner_label, request.instance_id
        )


def _join_error(current: Optional[str], new: str) -> str:
    return f"{current}; {new}" if current else new


def _fallback_reward_payload(
    instance_id: str, error_msg: Optional[str]
) -> Dict[str, Any]:
    return {
        "instance_id": instance_id,
        "resolved": False,
        "resolved_status": "error",
        "reward": 0,
        "error": error_msg,
    }


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _write_fallback_result(
    trial_dir: Path,
    reward_payload: Dict[str, Any],
    metadata: Dict[str, Any],
    error_msg: Optional[str],
) -> None:
    fallback_reward = dict(reward_payload)
    fallback_reward.setdefault("reward", 0)
    fallback_reward["error"] = fallback_reward.get("error") or error_msg
    fallback_metadata = dict(metadata)
    fallback_metadata["error"] = fallback_metadata.get("error") or error_msg
    _safe_write_json(trial_dir / "reward.json", fallback_reward)
    _safe_write_json(trial_dir / "metadata.json", fallback_metadata)
    _safe_write_json(
        trial_dir / "trajectory.json",
        {
            "success": False,
            "error": error_msg,
            "messages": [],
        },
    )


def run_task(
    request: TaskRunRequest,
    adapter: TaskAdapter,
) -> Dict[str, Any]:
    ensure_logging()
    started = time.time()
    trial_dir = Path(request.output_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    env_obj = None
    task: Optional[TaskSpec] = None
    agent_result = None
    eval_payload: Dict[str, Any] = {}
    eval_output: Optional[str] = None
    error_msg: Optional[str] = None
    agent_time = 0.0
    eval_time = 0.0
    tools_json = None

    try:
        task = adapter.prepare_task(request, trial_dir)
        with env_logging(trial_dir, adapter.env_logger_name):
            with trial_logging(trial_dir):
                logger.info(
                    "[%s] Starting environment: %s",
                    request.instance_id,
                    adapter.describe_environment(task, request),
                )
                env_obj = adapter.create_environment(task, request)
                adapter.start_environment(env_obj, task, request)
                adapter.setup_environment(env_obj, task, request)

                logger.info(
                    "[%s] Running %s", request.instance_id, adapter.runner_label
                )
                agent_start = time.time()
                agent = adapter.build_agent(env_obj, task, request, trial_dir)
                tools_json = adapter.get_tools_json(agent)
                agent_result = adapter.run_agent(agent, task, env_obj)
                adapter.after_agent_result(agent_result)
                agent_time = time.time() - agent_start

                logger.info("[%s] Running eval", request.instance_id)
                eval_start = time.time()
                with eval_logging(trial_dir, adapter.eval_logger_name):
                    eval_payload, eval_output = adapter.evaluate(
                        env_obj,
                        task,
                        request,
                        trial_dir,
                    )
                eval_time = time.time() - eval_start
    except Exception as exc:
        error_msg = str(exc)
        logger.exception(
            "%s run failed for %s", adapter.runner_label, request.instance_id
        )
    finally:
        if env_obj:
            _stop_environment(adapter, env_obj, task, request)

    try:
        reward_payload = adapter.build_reward_payload(
            request.instance_id,
            eval_payload,
            error_msg,
        )
    except Exception as exc:
        error_msg = _join_error(error_msg, f"reward build failed: {exc}")
        logger.exception(
            "%s reward build failed for %s", adapter.runner_label, request.instance_id
        )
        reward_payload = _fallback_reward_payload(request.instance_id, error_msg)

    metadata = build_metadata(
        request.instance_id,
        request.env_type,
        eval_payload,
        error_msg,
        trial_dir,
        eval_output,
        agent_result,
        reward_payload,
    )
    if task is not None:
        metadata.setdefault("task_id", task.id)
        metadata.setdefault("task_kind", task.kind)
    try:
        metadata = adapter.update_metadata(
            metadata,
            task,
            agent_result,
            eval_payload,
            error_msg,
        )
    except Exception as exc:
        error_msg = _join_error(error_msg, f"metadata update failed: {exc}")
        logger.exception(
            "%s metadata update failed for %s",
            adapter.runner_label,
            request.instance_id,
        )
        metadata["error"] = error_msg

    try:
        adapter.write_result(
            trial_dir,
            request.instance_id,
            request.model_name,
            request.base_url,
            request.env_type,
            agent_result,
            tools_json,
            reward_payload,
            eval_output,
            started,
            metadata,
        )
    except Exception as exc:
        error_msg = _join_error(error_msg, f"result write failed: {exc}")
        logger.exception(
            "%s result write failed for %s", adapter.runner_label, request.instance_id
        )
        metadata["error"] = error_msg
        reward_payload["error"] = reward_payload.get("error") or error_msg
        try:
            _write_fallback_result(trial_dir, reward_payload, metadata, error_msg)
        except Exception:
            logger.exception(
                "%s fallback result write failed for %s",
                adapter.runner_label,
                request.instance_id,
            )

    messages = _agent_messages(agent_result)
    total_time = time.time() - started
    try:
        agent_metrics = adapter.build_agent_metrics(
            messages,
            agent_time,
            eval_time,
            total_time,
            agent_result,
            eval_payload,
        )
    except Exception as exc:
        error_msg = _join_error(error_msg, f"metrics build failed: {exc}")
        logger.exception(
            "%s metrics build failed for %s", adapter.runner_label, request.instance_id
        )
        metadata["error"] = error_msg
        agent_metrics = build_agent_metrics(messages, agent_time, eval_time, total_time)

    try:
        exit_status = adapter.build_exit_status(error_msg, agent_result, eval_payload)
    except Exception as exc:
        error_msg = _join_error(error_msg, f"exit status build failed: {exc}")
        logger.exception(
            "%s exit status build failed for %s",
            adapter.runner_label,
            request.instance_id,
        )
        metadata["error"] = error_msg
        exit_status = "error"

    gc.collect()
    return {
        "reward": reward_payload.get("reward", 0),
        "messages": messages,
        "exit_status": exit_status,
        "agent_metrics": agent_metrics,
        "metadata": metadata,
        "tools": tools_json,
    }
