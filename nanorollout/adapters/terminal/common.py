"""Terminal Bench runner helpers."""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from nanorollout.runner import (
    build_reward_payload as _shared_build_reward_payload,
)

logger = logging.getLogger(__name__)
ENV_LOGGER_NAME = "nanorollout.envs.tools"
DEFAULT_TERMINAL_BENCH_REPO_URL = (
    "https://github.com/harbor-framework/terminal-bench-2.git"
)


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
    del create_timeout, step_timeout, eval_timeout
    env_timeout = env_timeout or 120
    if env_type == "docker":
        from nanorollout.envs.shell_env.docker import DockerEnvironment

        return DockerEnvironment(
            image=image,
            workspace_dir=workspace_dir,
            timeout=env_timeout,
        )
    if env_type == "modal":
        from nanorollout.envs.shell_env.modal import ModalEnvironment

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
    instance_id: str,
    eval_payload: Dict[str, Any],
    error_msg: Optional[str],
) -> Dict[str, Any]:
    return _shared_build_reward_payload(
        instance_id,
        eval_payload,
        error_msg,
        default_status="unresolved",
    )


def _parse_workdir_from_dockerfile(dockerfile: Path) -> str:
    if not dockerfile.is_file():
        return "/"
    workdir = "/"
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*WORKDIR\s+(.+)", line)
        if match:
            workdir = match.group(1).strip()
    return workdir


def _parse_size_to_mb(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    value = value.strip().upper()
    if value.endswith("G"):
        return int(float(value[:-1]) * 1024)
    if value.endswith("M"):
        return int(float(value[:-1]))
    return int(value)


def _parse_task_toml_dir(task_dir: Path) -> dict[str, Any]:
    with open(task_dir / "task.toml", "rb") as f:
        config = tomllib.load(f)
    agent_cfg = config.get("agent", {})
    verifier_cfg = config.get("verifier", {})
    env_cfg = config.get("environment", {})
    meta = config.get("metadata", {})
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"

    return {
        "instance_id": task_dir.name,
        "task_format": "tb2",
        "instruction": (task_dir / "instruction.md").read_text(encoding="utf-8"),
        "docker_image": env_cfg.get("docker_image"),
        "dockerfile_dir": str(env_dir) if dockerfile.is_file() else None,
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
    with open(task_dir / "task.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    dockerfile = task_dir / "Dockerfile"
    run_tests = task_dir / "run-tests.sh"
    workspace_dir = _parse_workdir_from_dockerfile(dockerfile)
    if workspace_dir == "/":
        workspace_dir = "/app"

    return {
        "instance_id": task_dir.name,
        "task_format": "tb_legacy",
        "instruction": config.get("instruction", ""),
        "docker_image": None,
        "dockerfile_dir": str(task_dir) if dockerfile.is_file() else None,
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
    if repo_dir and not _is_remote_repo(repo_dir):
        path = Path(repo_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Repository directory not found: {repo_dir}")
        return path

    source_url = repo_dir if repo_dir and _is_remote_repo(repo_dir) else repo_url
    cache_root = Path(
        os.environ.get("TB_REPO_CACHE_DIR", "~/.cache/nanorollout/terminalbench")
    ).expanduser()
    repo_path = cache_root / re.sub(r"[^a-zA-Z0-9._-]", "-", source_url).strip("-")

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
    existing = instance.get("docker_image")
    if existing:
        return existing

    dockerfile_dir = instance.get("dockerfile_dir")
    if not dockerfile_dir:
        raise ValueError(
            f"Instance {instance.get('instance_id')} has no docker_image and no dockerfile_dir"
        )

    if env_type in ("modal", "daytona"):
        instance["docker_image"] = dockerfile_dir
        return dockerfile_dir
    if env_type not in ("docker", "enroot"):
        raise ValueError(
            f"Building from Dockerfile is not supported for env_type={env_type}"
        )
    if shutil.which("docker") is None:
        raise RuntimeError(
            "docker CLI not found on PATH; cannot build image from Dockerfile"
        )

    instance_id = str(instance.get("instance_id", "unknown"))
    safe_id = re.sub(r"[^a-z0-9._-]", "-", instance_id.lower()).strip("-") or "task"
    tag = f"tinyflow-tb-{safe_id}:latest"
    build_timeout = int(instance.get("build_timeout_sec") or 1800)

    logger.info(
        "Building image %s from %s (timeout=%ss)", tag, dockerfile_dir, build_timeout
    )
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
    instance: dict[str, Any],
    extra_args: dict[str, Any],
) -> tuple[int, int]:
    multiplier = extra_args.get("timeout_multiplier", 1.0)

    global_agent_cap = extra_args.get("step_timeout", float("inf"))
    task_agent = instance.get("agent_timeout_sec", 600.0)
    agent_timeout = (
        task_agent * multiplier
        if global_agent_cap == -1
        else min(task_agent, global_agent_cap) * multiplier
    )

    global_eval_cap = extra_args.get("eval_timeout", float("inf"))
    task_eval = instance.get("eval_timeout_sec", 600.0)
    eval_timeout = (
        task_eval * multiplier
        if global_eval_cap == -1
        else min(task_eval, global_eval_cap) * multiplier
    )

    return max(1, math.ceil(agent_timeout)), max(1, math.ceil(eval_timeout))


def load_tb_instances(
    repo_dir: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str, Any]]:
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
        logger.info(
            "Loaded %d Terminal Bench instances from %s", len(instances), repo_path
        )
    return instances


def resolve_tb_instance(
    instance_id: str,
    repo_dir: str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    del logger
    task_dir = Path(repo_dir).resolve() / instance_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    return _parse_task_dir(task_dir)
