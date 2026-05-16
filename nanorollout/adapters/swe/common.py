"""SWE task runner helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from nanorollout.runner import (
    build_reward_payload as _shared_build_reward_payload,
    write_json,
    write_task_artifacts,
)

ENV_LOGGER_NAME = "nanorollout.envs.tools"


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
        from nanorollout.envs.shell_env.docker import DockerEnvironment

        return DockerEnvironment(
            image=image,
            workspace_dir=workspace_dir,
            timeout=env_timeout,
        )
    if env_type == "enroot":
        from nanorollout.envs.shell_env.enroot import EnrootEnvironment

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
        from nanorollout.envs.shell_env.modal import ModalEnvironment

        return ModalEnvironment(
            image=image,
            instance=instance,
            workspace_dir=workspace_dir,
            timeout=env_timeout,
        )
    raise ValueError(f"Unsupported environment type: {env_type}")


def build_reward_payload(
    instance_id: str,
    eval_payload: Dict[str, Any],
    error_msg: Optional[str],
) -> Dict[str, Any]:
    payload = _shared_build_reward_payload(
        instance_id,
        eval_payload,
        error_msg,
        default_status="unknown",
    )
    payload["status_map"] = eval_payload.get("status_map", {})
    payload["report"] = eval_payload.get("report", {})
    return payload


def _default_tests_status() -> dict[str, dict[str, list[str]]]:
    return {
        "FAIL_TO_PASS": {"success": [], "failure": []},
        "PASS_TO_PASS": {"success": [], "failure": []},
        "FAIL_TO_FAIL": {"success": [], "failure": []},
        "PASS_TO_FAIL": {"success": [], "failure": []},
    }


def _write_swe_report(
    trial_dir: Path,
    instance_id: str,
    patch: str,
    reward_payload: Dict[str, Any],
) -> None:
    report = {
        instance_id: {
            "patch_is_None": patch == "",
            "patch_exists": patch != "",
            "patch_successfully_applied": bool(reward_payload.get("report")),
            "resolved": reward_payload.get("resolved", False),
            "tests_status": reward_payload.get("report") or _default_tests_status(),
        }
    }
    write_json(trial_dir / "report.json", report)


def write_artifacts(
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
    patch = getattr(agent_result, "patch", "") if agent_result else ""
    (trial_dir / "model.patch").write_text(patch, encoding="utf-8")
    _write_swe_report(trial_dir, instance_id, patch, reward_payload)
