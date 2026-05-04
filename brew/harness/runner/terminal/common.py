import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

# TODO: use AgentConfig for terminal
from brew.harness.agents.swe.base import AgentConfig


logger = logging.getLogger(__name__)
ENV_LOGGER_NAME = "brew.envs.tools"
DEFAULT_TERMINAL_BENCH_REPO_URL = "https://github.com/harbor-framework/terminal-bench-2.git"


class ThreadLogFilter(logging.Filter):
    """Filter log records to a single thread."""

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self._thread_id


@contextmanager
def trial_logging(trial_dir: Path) -> Iterator[Path]:
    """Attach a per-thread log file handler for the current trial."""
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
def env_logging(run_dir: Path, filename: str = "env.log") -> Iterator[Path]:
    """Attach a log file handler for environment tool execution logs."""
    log_path = run_dir / filename
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    env_logger = logging.getLogger(ENV_LOGGER_NAME)
    prev_level = env_logger.level
    prev_propagate = env_logger.propagate
    env_logger.setLevel(logging.INFO)
    env_logger.propagate = False
    env_logger.addHandler(handler)
    try:
        yield log_path
    finally:
        env_logger.removeHandler(handler)
        handler.close()
        env_logger.setLevel(prev_level)
        env_logger.propagate = prev_propagate


@contextmanager
def eval_logging(trial_dir: Path, filename: str = "eval.log") -> Iterator[Path]:
    """Attach a per-thread log file handler for the eval phase."""
    log_path = trial_dir / filename
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def write_trajectory(trial_dir: Path, payload: dict[str, Any]) -> Path:
    path = trial_dir / "trajectory.json"
    _write_json(path, payload)
    return path


def create_environment(
    env_type: str,
    instance: Dict[str, Any],
    image: str,
    workspace_dir: str,
    env_timeout: Optional[int] = None,
    create_timeout: Optional[int] = None,
    step_timeout: Optional[int] = None,
    eval_timeout: Optional[int] = None,
):
    create_timeout = create_timeout or 600
    step_timeout = step_timeout or 600
    eval_timeout = eval_timeout or 600
    env_timeout = env_timeout or 120
    if env_type == "docker":
        from brew.envs.shell_env.docker import DockerEnvironment

        return DockerEnvironment(
            image=image,
            workspace_dir=workspace_dir,
            timeout=env_timeout,
        )
    if env_type == "modal":
        from brew.envs.shell_env.modal import ModalEnvironment

        return ModalEnvironment(
            image=image,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=env_timeout,
            cpu=instance.get("cpus", 0.5),
            memory_mb=instance.get("memory_mb", 128),
        )
    raise ValueError(f"Unsupported environment type: {env_type}")


def build_reward_payload(
    instance_id: str, eval_payload: Dict[str, Any], error_msg: Optional[str]
) -> Dict[str, Any]:
    return {
        "instance_id": instance_id,
        "resolved": eval_payload.get("resolved", False),
        "resolved_status": eval_payload.get("resolved_status", "unresolved"),
        "reward": eval_payload.get("reward", 0),
        "eval_exit_code": eval_payload.get("eval_exit_code"),
        "error": eval_payload.get("error") or error_msg,
        "status_map": eval_payload.get("status_map", {}),
        "report": eval_payload.get("report", {}),
    }


def _ensure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        root.setLevel(level)


def _build_agent_config(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_iterations: Optional[int],
    sampling_params: Dict[str, Any],
    default_temperature: float = 0.6,
    default_top_p: float = 0.95,
) -> AgentConfig:
    if isinstance(sampling_params, str):
        try:
            sampling_params = json.loads(sampling_params)
        except json.JSONDecodeError:
            sampling_params = {}
    if not isinstance(sampling_params, dict):
        sampling_params = {}
    max_iterations = max_iterations or 100
    temperature = sampling_params.get("temperature", default_temperature)
    top_p = sampling_params.get("top_p", default_top_p)
    extra_body = sampling_params.get("extra_body", {})
    max_tokens = sampling_params.get("max_tokens", 4096)
    return AgentConfig(
        model=model,
        max_iterations=max_iterations,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=base_url,
    )


def _write_artifacts(
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
        "success": bool(agent_result and agent_result.success),
        "message": agent_result.message if agent_result else "",
        "iterations": agent_result.iterations if agent_result else 0,
        "error": agent_result.error if agent_result else metadata.get("error"),
        "messages": agent_result.history if agent_result else [],
        "llm_metrics": agent_result.llm_metrics if agent_result else [],
        "llm_cost_total": agent_result.llm_cost_total if agent_result else 0.0,
        "wall_time_sec": round(time.time() - started, 2),
        "tools": tools_json,
    }
    if exit_status is not None:
        traj_payload["exit_status"] = exit_status
    write_trajectory(trial_dir, traj_payload)

    reward_path = trial_dir / "reward.json"
    reward_path.write_text(
        json.dumps(reward_payload_to_write, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if eval_output is not None:
        eval_output_path = trial_dir / "eval_output.txt"
        eval_output_path.write_text(eval_output, encoding="utf-8")


def _build_metadata(
    instance_id: str,
    env_type: str,
    eval_payload: Dict[str, Any],
    error_msg: Optional[str],
    trial_dir: Optional[Path],
    eval_output: Optional[str],
    agent_result: Any,
    reward_payload: Dict[str, Any],
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
        "exit_status": exit_status or ("setup_error" if agent_result is None else None),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _build_agent_metrics(
    messages: list[Dict[str, Any]],
    agent_time: float,
    eval_time: float,
    total_time: float,
) -> Dict[str, Any]:
    turns = sum(1 for msg in messages if msg.get("role") == "assistant")
    tool_calls = _count_tool_calls(messages)
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "model_query_time_sum": 0.0,
        "env_execution_time_sum": 0.0,
        "eval_time": eval_time,
        "agent_run_time": agent_time,
        "total_time": total_time,
    }


def _count_tool_calls(messages: list[Dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            count += len(tool_calls)
    return count


def _resolve_exit_status(eval_payload: Dict[str, Any]) -> str:
    if eval_payload.get("resolved"):
        return "Resolved"
    return "Completed"


def _parse_workdir_from_dockerfile(dockerfile: Path) -> str:
    """Extract the last WORKDIR from a Dockerfile, defaulting to '/'."""
    if not dockerfile.is_file():
        return "/"
    workdir = "/"
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*WORKDIR\s+(.+)", line)
        if m:
            workdir = m.group(1).strip()
    return workdir


def _parse_size_to_mb(value: str | int | float) -> int:
    """Parse a size string like '2G', '512M' into megabytes."""
    if isinstance(value, (int, float)):
        return int(value)
    value = value.strip().upper()
    if value.endswith("G"):
        return int(float(value[:-1]) * 1024)
    if value.endswith("M"):
        return int(float(value[:-1]))
    return int(value)


def _parse_task_toml_dir(task_dir: Path) -> dict[str, Any]:
    """TB2 / Harbor format: task.toml + instruction.md + environment/Dockerfile."""
    with open(task_dir / "task.toml", "rb") as f:
        config = tomllib.load(f)
    agent_cfg = config.get("agent", {})
    verifier_cfg = config.get("verifier", {})
    env_cfg = config.get("environment", {})
    meta = config.get("metadata", {})
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"
    dockerfile_dir = str(env_dir) if dockerfile.is_file() else None

    return {
        "instance_id": task_dir.name,
        "task_format": "tb2",
        "instruction": (task_dir / "instruction.md").read_text(encoding="utf-8"),
        "docker_image": env_cfg.get("docker_image"),
        "dockerfile_dir": dockerfile_dir,
        "workspace_dir": _parse_workdir_from_dockerfile(dockerfile),
        "agent_timeout_sec": agent_cfg.get("timeout_sec", 600.0),
        "eval_timeout_sec": verifier_cfg.get("timeout_sec", 600.0),
        "build_timeout_sec": env_cfg.get("build_timeout_sec", 600.0),
        "cpus": env_cfg.get("cpus", 1),
        "memory_mb": _parse_size_to_mb(env_cfg.get("memory", "2G")),
        "storage_mb": _parse_size_to_mb(env_cfg.get("storage", "10G")),
        "difficulty": meta.get("difficulty", "unknown"),
        "category": meta.get("category", "unknown"),
        "tags": meta.get("tags", []),
    }


def _parse_task_yaml_dir(task_dir: Path) -> dict[str, Any]:
    """Terminal-Bench legacy format: task.yaml + Dockerfile + tests/ at task root."""
    with open(task_dir / "task.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    dockerfile = task_dir / "Dockerfile"
    dockerfile_dir = str(task_dir) if dockerfile.is_file() else None
    run_tests = task_dir / "run-tests.sh"

    # Legacy Terminal-Bench tasks inherit WORKDIR /app from the t-bench base
    # image. Our parser only scans the task's own Dockerfile, so fall back to
    # /app (not /) when no explicit WORKDIR is declared — matches TB convention
    # and avoids the `$PWD=/` guard in run-tests.sh.
    workspace_dir = _parse_workdir_from_dockerfile(dockerfile)
    if workspace_dir == "/":
        workspace_dir = "/app"

    return {
        "instance_id": task_dir.name,
        "task_format": "tb_legacy",
        "instruction": config.get("instruction", ""),
        "docker_image": None,
        "dockerfile_dir": dockerfile_dir,
        "workspace_dir": workspace_dir,
        "agent_timeout_sec": float(config.get("max_agent_timeout_sec", 600.0)),
        "eval_timeout_sec": float(config.get("max_test_timeout_sec", 600.0)),
        "build_timeout_sec": 600.0,
        "cpus": 1,
        "memory_mb": 2048,
        "storage_mb": 10240,
        "difficulty": config.get("difficulty", "unknown"),
        "category": config.get("category", "unknown"),
        "tags": config.get("tags", []),
        "parser_name": config.get("parser_name"),
        "run_tests_script": str(run_tests) if run_tests.is_file() else None,
    }


def _parse_task_dir(task_dir: Path) -> dict[str, Any]:
    """Parse a task directory, autodetecting tb2 (task.toml) vs legacy (task.yaml)."""
    if (task_dir / "task.toml").is_file():
        return _parse_task_toml_dir(task_dir)
    if (task_dir / "task.yaml").is_file():
        return _parse_task_yaml_dir(task_dir)
    raise FileNotFoundError(f"No task.toml or task.yaml found in {task_dir}")


def _is_remote_repo(value: str) -> bool:
    return value.startswith(("http://", "https://", "git@")) or value.endswith(".git")


def resolve_tb_repo_dir(
    repo_dir: Optional[str] = None,
    *,
    repo_url: str = DEFAULT_TERMINAL_BENCH_REPO_URL,
    revision: Optional[str] = None,
    refresh: bool = False,
) -> Path:
    """Resolve a Terminal Bench repo, cloning the default remote into a cache if needed."""
    if repo_dir and not _is_remote_repo(repo_dir):
        path = Path(repo_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Repository directory not found: {repo_dir}")
        return path

    source_url = repo_dir if repo_dir and _is_remote_repo(repo_dir) else repo_url
    cache_root = Path(
        os.environ.get("TB_REPO_CACHE_DIR", "~/.cache/brew/terminalbench")
    ).expanduser()
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", source_url).strip("-")
    repo_path = cache_root / safe_name

    if not repo_path.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        clone_cmd = ["git", "clone", "--depth", "1"]
        if revision:
            clone_cmd.extend(["--branch", revision])
        clone_cmd.extend([source_url, str(repo_path)])
        logger.info("Cloning Terminal Bench repo %s into %s", source_url, repo_path)
        subprocess.run(
            clone_cmd,
            check=True,
            timeout=600,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    elif refresh:
        logger.info("Refreshing Terminal Bench repo cache at %s", repo_path)
        subprocess.run(
            ["git", "-C", str(repo_path), "fetch", "--depth", "1", "origin"],
            check=True,
            timeout=600,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    if revision and repo_path.exists():
        subprocess.run(
            ["git", "-C", str(repo_path), "checkout", revision],
            check=True,
            timeout=120,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    return repo_path.resolve()


def ensure_tb_image(instance: dict[str, Any], env_type: str = "docker") -> str:
    """Return an image reference for the instance, building from a Dockerfile if needed.

    If the instance already names a pre-built image, return it. Otherwise build the
    Dockerfile at ``dockerfile_dir`` with ``docker build`` and return the tag.
    Mutates ``instance`` so subsequent calls are cheap.
    """
    existing = instance.get("docker_image")
    if existing:
        return existing

    dockerfile_dir = instance.get("dockerfile_dir")
    if not dockerfile_dir:
        raise ValueError(
            f"Instance {instance.get('instance_id')} has no docker_image and no dockerfile_dir"
        )

    if env_type in ("modal", "daytona"):
        # Modal and Daytona build Dockerfiles natively at sandbox start, so
        # pass the build-context directory straight through.
        instance["docker_image"] = dockerfile_dir
        return dockerfile_dir

    if env_type not in ("docker", "enroot"):
        raise ValueError(
            f"ensure_tb_image: building from Dockerfile is not supported for env_type={env_type}"
        )
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI not found on PATH; cannot build image from Dockerfile")

    instance_id = str(instance.get("instance_id", "unknown"))
    safe_id = re.sub(r"[^a-z0-9._-]", "-", instance_id.lower()).strip("-") or "task"
    tag = f"tinyflow-tb-{safe_id}:latest"
    build_timeout = int(instance.get("build_timeout_sec") or 1800)

    logger.info("Building image %s from %s (timeout=%ss)", tag, dockerfile_dir, build_timeout)
    try:
        subprocess.run(
            ["docker", "build", "-t", tag, dockerfile_dir],
            check=True,
            timeout=build_timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        raise RuntimeError(f"docker build failed for {instance_id}: {output}") from exc

    instance["docker_image"] = tag
    return tag


def resolve_tb_timeouts(
    instance: dict[str, Any], extra_args: dict[str, Any]
) -> tuple[int, int]:
    """Resolve per-task Terminal Bench timeouts aligned with Harbor.

    Special global cap values:
    - ``-1`` means use the task value directly.
    - positive values cap the task value with ``min(task_value, cap)``.
    - omitted / infinity means no effective cap.
    """
    multiplier = extra_args.get("timeout_multiplier", 1.0)

    global_agent_cap = extra_args.get("step_timeout", float("inf"))
    task_agent = instance.get("agent_timeout_sec", 600.0)
    if global_agent_cap == -1:
        agent_timeout = task_agent * multiplier
    else:
        agent_timeout = min(task_agent, global_agent_cap) * multiplier

    global_eval_cap = extra_args.get("eval_timeout", float("inf"))
    task_eval = instance.get("eval_timeout_sec", 600.0)
    if global_eval_cap == -1:
        eval_timeout = task_eval * multiplier
    else:
        eval_timeout = min(task_eval, global_eval_cap) * multiplier

    return max(1, math.ceil(agent_timeout)), max(1, math.ceil(eval_timeout))


def load_tb_instances(
    repo_dir: str, *, logger: Optional[logging.Logger] = None
) -> list[dict[str, Any]]:
    """Load all task instances from a Terminal Bench repository directory."""
    repo_path = Path(repo_dir).resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Repository directory not found: {repo_dir}")

    instances: list[dict[str, Any]] = []
    for task_dir in sorted(repo_path.iterdir()):
        if not task_dir.is_dir():
            continue
        try:
            instances.append(_parse_task_dir(task_dir))
        except Exception as exc:
            if logger:
                logger.debug("Skipping %s: %s", task_dir.name, exc)
    if logger:
        logger.info("Loaded %d Terminal Bench instances from %s", len(instances), repo_path)
    return instances


def resolve_tb_instance(
    instance_id: str,
    repo_dir: str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Load a single task instance by ID from the repository."""
    task_dir = Path(repo_dir).resolve() / instance_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    return _parse_task_dir(task_dir)