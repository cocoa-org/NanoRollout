import json
import logging
from typing import Any, Dict, Optional

from datasets import load_dataset

from .common import DATASET_MAPPING


def _normalize_r2e_gym_instance(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an R2E-Gym row to match the common instance schema."""

    if "instance_id" not in row:
        repo = row.get("repo_name", "unknown")
        commit = row.get("commit_hash", "unknown")
        row["instance_id"] = f"{repo}__{commit}"

    parsed_commit_content = json.loads(row.get("parsed_commit_content", "{}"))
    old_commit_hash = parsed_commit_content.get("old_commit_hash")
    if not old_commit_hash:
        raise ValueError("R2E-Gym: old_commit_hash not found in parsed_commit_content")

    # old_commit_hash in R2E-Gym dataset is the base commit
    row["base_commit"] = old_commit_hash

    # TODO: parse `parsed_commit_content` into canonical patch
    row["patch"] = row.get("parsed_commit_content", "")
    return row


def load_instances(
    subset: str, split: str, *, logger: Optional[logging.Logger] = None
) -> list[dict[str, Any]]:
    dataset_name = DATASET_MAPPING.get(subset, subset)
    if logger:
        logger.info("Loading dataset %s split %s", dataset_name, split)
    rows = list(load_dataset(dataset_name, split=split))
    if subset in DATASET_MAPPING and DATASET_MAPPING[subset] == "R2E-Gym/R2E-Gym-V1":
        rows = [_normalize_r2e_gym_instance(row) for row in rows]
    return rows


def select_instance(
    instances: list[dict[str, Any]],
    instance_id: Optional[str],
    index: Optional[int],
    *,
    sort_by_instance_id: bool = False,
) -> dict[str, Any]:
    if instance_id:
        for instance in instances:
            if instance.get("instance_id") == instance_id:
                return dict(instance)
        raise ValueError(f"Instance id not found: {instance_id}")
    if index is None:
        raise ValueError("Missing instance_id or index for dataset selection")
    candidates = instances
    if sort_by_instance_id:
        candidates = sorted(candidates, key=lambda x: x["instance_id"])
    index_value = int(index)
    if index_value < 0 or index_value >= len(candidates):
        raise IndexError(
            f"Index {index_value} is out of range for {len(candidates)} instances"
        )
    return dict(candidates[index_value])


def resolve_instance(
    instance_id: str,
    subset: str,
    split: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    instances = load_instances(subset, split, logger=logger)
    instance = select_instance(
        instances=instances, instance_id=instance_id, index=None, sort_by_instance_id=False
    )
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a dict")
    if instance_id and instance.get("instance_id") != instance_id:
        raise ValueError(f"Instance id mismatch: {instance.get('instance_id')} != {instance_id}")
    return instance


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
    if env_type == "enroot":
        from brew.envs.shell_env.enroot import EnrootEnvironment

        return EnrootEnvironment(
            image=image,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=env_timeout,
            create_timeout=create_timeout,
            step_timeout=step_timeout,
            eval_timeout=eval_timeout,
        )
    if env_type == "modal":
        from brew.envs.shell_env.modal import ModalEnvironment

        return ModalEnvironment(
            image=image,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=env_timeout,
        )
    raise ValueError(f"Unsupported environment type: {env_type}")


def build_eval_payload(instance: Dict[str, Any], output: str) -> Dict[str, Any]:
    from brew.eval.swebench.constants import (
        END_TEST_OUTPUT,
        START_TEST_OUTPUT,
        ResolvedStatus,
    )
    from brew.eval.swebench.grading import get_eval_report

    if START_TEST_OUTPUT not in output or END_TEST_OUTPUT not in output:
        return {
            "resolved": False,
            "resolved_status": ResolvedStatus.NO.value,
            "reward": 0,
            "error": "missing test output markers",
            "status_map": {},
            "report": {},
        }

    try:
        report = get_eval_report(instance, output)
    except Exception as exc:
        return {
            "resolved": False,
            "resolved_status": ResolvedStatus.NO.value,
            "reward": 0,
            "error": str(exc),
            "status_map": {},
            "report": {},
        }

    resolved_status = report.get("resolved_status", ResolvedStatus.NO.value)
    resolved = resolved_status == ResolvedStatus.FULL.value
    reward = 1 if resolved else 0

    return {
        "resolved": resolved,
        "resolved_status": resolved_status,
        "reward": reward,
        "status_map": report.get("test_status_map", {}),
        "report": report.get("metrics", {}),
    }


def build_reward_payload(
    instance_id: str, eval_payload: Dict[str, Any], error_msg: Optional[str]
) -> Dict[str, Any]:
    return {
        "instance_id": instance_id,
        "resolved": eval_payload.get("resolved", False),
        "resolved_status": eval_payload.get("resolved_status", "NO"),
        "reward": eval_payload.get("reward", 0),
        "eval_exit_code": eval_payload.get("eval_exit_code"),
        "error": eval_payload.get("error") or error_msg,
        "status_map": eval_payload.get("status_map", {}),
        "report": eval_payload.get("report", {}),
    }
