"""
Artifact helpers for saving runs and trials.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
ENV_LOGGER_NAME = "oh_core.env.tools"


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
