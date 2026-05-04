"""Shared helpers for SWE runner entry points."""

import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brew.harness.agents.swe.base import AgentConfig

DEFAULT_RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
ENV_LOGGER_NAME = "oh_core.env.tools"

logger = logging.getLogger(__name__)


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

    logger = logging.getLogger(ENV_LOGGER_NAME)
    prev_level = logger.level
    prev_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield log_path
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate


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


def ensure_run_dir(output_root: Path, run_id: Optional[str] = None) -> Path:
    if not run_id:
        run_id = time.strftime(DEFAULT_RUN_ID_FORMAT)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_trial_dir(run_dir: Path, task_id: str, trial_id: int) -> Path:
    trial_name = f"trial_{trial_id:04d}"
    trial_dir = run_dir / task_id / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    return trial_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def write_trajectory(trial_dir: Path, payload: dict[str, Any]) -> Path:
    path = trial_dir / "trajectory.json"
    _write_json(path, payload)
    return path


def write_patch(trial_dir: Path, patch: Optional[str]) -> Path:
    path = trial_dir / "model.patch"
    path.write_text(patch or "", encoding="utf-8")
    return path


def default_tests_status() -> dict[str, dict[str, list[str]]]:
    return {
        "FAIL_TO_PASS": {"success": [], "failure": []},
        "PASS_TO_PASS": {"success": [], "failure": []},
        "FAIL_TO_FAIL": {"success": [], "failure": []},
        "PASS_TO_FAIL": {"success": [], "failure": []},
    }


def build_report_payload(
    instance_id: str,
    patch: Optional[str],
    tests_status: Optional[dict[str, Any]],
    *,
    resolved: bool,
    patch_applied: bool,
) -> dict[str, Any]:
    patch_value = patch or ""
    tests_payload = tests_status if tests_status else default_tests_status()
    return {
        instance_id: {
            "patch_is_None": patch_value == "",
            "patch_exists": patch_value != "",
            "patch_successfully_applied": patch_applied,
            "resolved": resolved,
            "tests_status": tests_payload,
        }
    }


def write_report(trial_dir: Path, payload: dict[str, Any]) -> Path:
    path = trial_dir / "report.json"
    _write_json(path, payload)
    return path


DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "gym": "SWE-Gym/SWE-Gym",
    "rebench": "nebius/SWE-rebench",
    "smith-py": "SWE-bench/SWE-smith-py",
    "r2e-gym": "R2E-Gym/R2E-Gym-V1",
}


class NamingStrategy:
    SWE_BENCH = "swe_bench"
    SWE_GYM = "swe_gym"
    SWE_REBENCH = "swe_rebench"
    SWE_SMITH = "swe_smith"
    R2E_GYM = "r2e_gym"


def get_swebench_docker_image_name(
    instance: dict[str, Any],
    env_class: str,
    naming_strategy: str,
) -> str:
    # image_name = instance.get("image_name")
    # if image_name:
    #     return image_name
    instance_id = instance["instance_id"]
    if naming_strategy == NamingStrategy.SWE_GYM:
        id_docker_compatible = instance_id.replace("__", "_s_")
        namespace = "xingyaoww"
        dataset = "sweb"
        split = "eval"
    elif naming_strategy == NamingStrategy.SWE_REBENCH:
        id_docker_compatible = instance_id.replace("__", "_1776_")
        namespace = "swerebench"
        dataset = "sweb"
        split = "eval"
    elif naming_strategy == NamingStrategy.SWE_SMITH:
        id_docker_compatible = instance_id.split(".")[0] + "." + instance_id.split(".")[1]
        id_docker_compatible = id_docker_compatible.replace("__", "_1776_")
        namespace = "jyangballin"
        dataset = "swesmith"
        split = ""
    elif naming_strategy == NamingStrategy.R2E_GYM:
        # R2E-Gym uses repo_name and commit_hash (not instance_id) for image naming.
        # Image files: namanjain12+{repo}_final+{commit}.sqsh
        # at /mnt/weka/home/zhuojun.cheng/uda-org/dockers/swe/r2egym_images/
        repo_name = instance.get("repo_name", "")
        commit_hash = instance.get("commit_hash", "")
        namespace = "namanjain12"
        image_base = f"{namespace}+{repo_name}_final+{commit_hash}"
        if env_class == "enroot":
            return f"{image_base}.sqsh".lower()
        if env_class in ("docker", "modal"):
            return f"docker.io/{namespace}/{repo_name}_final:{commit_hash}".lower()
        raise ValueError(f"Unknown environment class: {env_class}")
    else:
        id_docker_compatible = instance_id.replace("__", "_1776_")
        namespace = "swebench"
        dataset = "sweb"
        split = "eval"
    split_str = f".{split}" if split else ""
    if env_class in ("docker", "modal"):
        return (
            f"docker.io/{namespace}/{dataset}{split_str}.x86_64.{id_docker_compatible}:latest"
        ).lower()
    if env_class == "enroot":
        return f"{namespace}+{dataset}{split_str}.x86_64.{id_docker_compatible}+latest.sqsh".lower()
    raise ValueError(f"Unknown environment class: {env_class}")


def _resolve_naming_strategy(subset: str) -> str:
    subset_lower = subset.lower()
    if "r2e" in subset_lower:
        return NamingStrategy.R2E_GYM
    if "gym" in subset_lower:
        return NamingStrategy.SWE_GYM
    if "rebench" in subset_lower:
        return NamingStrategy.SWE_REBENCH
    if "smith" in subset_lower:
        return NamingStrategy.SWE_SMITH
    return NamingStrategy.SWE_BENCH


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


def _run_eval(
    env_obj: Any,
    instance: Dict[str, Any],
    eval_timeout: Optional[int],
    workspace_dir: str,
    dataset: str = "gym",
) -> Tuple[Dict[str, Any], Optional[str]]:
    from .single_run import build_eval_payload

    instance_id = instance.get("instance_id", "unknown")
    eval_output = None
    if _resolve_naming_strategy(dataset) == NamingStrategy.R2E_GYM:
        from brew.eval.r2egym.grading import run_r2egym_eval

        return run_r2egym_eval(env_obj, instance, eval_timeout)

    try:
        logger.info("[%s] Generating test spec for eval", instance_id)
        from brew.eval.swebench.test_spec import make_test_spec

        test_spec = make_test_spec(instance, arch="x86_64", repo_directory=workspace_dir)
        logger.info(
            "[%s] Executing eval script (timeout=%ss)",
            instance_id,
            eval_timeout or 1800,
        )
        eval_result = env_obj.execute(
            test_spec.eval_script, timeout=eval_timeout or 1800
        )
        eval_output = eval_result.output
        logger.info(
            "[%s] Eval script finished with exit_code=%s, output_length=%d",
            instance_id,
            eval_result.exit_code,
            len(eval_output) if eval_output else 0,
        )
        eval_payload = build_eval_payload(instance, eval_output)
        eval_payload["eval_exit_code"] = eval_result.exit_code
        logger.info(
            "[%s] Eval result: resolved=%s, resolved_status=%s, reward=%s",
            instance_id,
            eval_payload.get("resolved"),
            eval_payload.get("resolved_status"),
            eval_payload.get("reward"),
        )
        return eval_payload, eval_output
    except Exception as exc:
        logger.exception("[%s] Eval failed with error: %s", instance_id, exc)
        return (
            {
                "resolved": False,
                "resolved_status": "NO",
                "reward": 0,
                "error": str(exc),
                "status_map": {},
                "report": {},
            },
            eval_output,
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
    write_patch(trial_dir, agent_result.patch if agent_result else "")

    reward_path = trial_dir / "reward.json"
    reward_path.write_text(
        json.dumps(reward_payload_to_write, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    report_payload = build_report_payload(
        instance_id,
        agent_result.patch if agent_result else "",
        reward_payload_to_write.get("report") or None,
        resolved=reward_payload_to_write.get("resolved", False),
        patch_applied=bool(reward_payload_to_write.get("report")),
    )
    write_report(trial_dir, report_payload)

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
        "exit_status": agent_result.exit_status if agent_result is not None else "setup_error",
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
    if eval_payload.get("resolved_status"):
        return str(eval_payload.get("resolved_status"))
    return "Completed"
