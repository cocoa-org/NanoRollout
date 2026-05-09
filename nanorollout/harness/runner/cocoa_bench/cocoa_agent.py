"""Cocoa-Bench runner backed by NanoRollout's in-repo Cocoa agent/env."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from nanorollout.harness.runner.cocoa_bench.common import (
    _attach_trial_log,
    _build_cocoa_config,
    _build_metadata,
    _build_response,
    _build_reward_payload,
    _coerce_bool,
    _detect_encrypted_task,
    _ensure_logging,
    _load_task,
    _parse_sampling_params,
    _resolve_task_root,
    _write_artifacts,
    _write_json,
    _allocate_port,
)

logger = logging.getLogger(__name__)


def run_cocoa_agent(
    instance_id: str,
    output_dir: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    env_type: str = "docker",
    sampling_params: Optional[object] = None,
    extra_args: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Run a single CocoaBench task via the in-repo Cocoa agent."""

    extra_args = dict(extra_args or {})
    log_level_name = str(extra_args.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    _ensure_logging(log_level)
    started = time.time()
    env_type = env_type or "docker"
    sampling_params_dict = _parse_sampling_params(sampling_params)

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trial_log_path = output_root / "trial.log"

    result: dict[str, Any] = {}
    error_msg: Optional[str] = None
    tasks_dir = output_root
    task_dir = output_root
    config_path = output_root / "cocoa_config.json"

    with _attach_trial_log(trial_log_path, log_level):
        logger.info("[%s] Writing Cocoa trial log to %s", instance_id, trial_log_path)
        try:
            tasks_dir, task_dir = _resolve_task_root(instance_id, extra_args)
            encrypted_task = _detect_encrypted_task(task_dir)
            preferred_port = extra_args.get("docker_port")
            docker_port = _allocate_port(int(preferred_port) if preferred_port is not None else None)

            config = _build_cocoa_config(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                env_type=env_type,
                sampling_params=sampling_params_dict,
                extra_args=extra_args,
                encrypted_task=encrypted_task,
                docker_port=docker_port,
            )
            _write_json(config_path, config)
            from nanorollout.envs.cocoa_env import setup_logging
            from nanorollout.harness.agents.cocoa import CocoaAgent

            setup_logging(
                str(config.get("log_level", log_level_name)),
                log_file=str(trial_log_path),
            )

            task = _load_task(
                task_dir,
                _coerce_bool(config.get("use_encrypted_tasks"), default=encrypted_task),
            )
            agent = CocoaAgent(config)
            wait_time = int(
                extra_args.get("create_timeout")
                or extra_args.get("env_timeout")
                or 30
            )

            logger.info("[%s] Running CocoaAgent task from %s", instance_id, task_dir)
            try:
                agent.setup_environment(task, wait_time=wait_time)
                result = agent.run_task(task)
                eval_result = agent.run_eval(task, result)
                if eval_result is not None:
                    result["eval"] = eval_result
            finally:
                try:
                    agent.cleanup_environment()
                except Exception as cleanup_exc:
                    logger.exception("CocoaAgent cleanup failed for %s", instance_id)
                    if error_msg is None:
                        error_msg = f"Cleanup failed: {cleanup_exc}"
        except Exception as exc:
            error_msg = str(exc)
            logger.exception("CocoaAgent run failed for %s", instance_id)

    reward_payload = _build_reward_payload(instance_id, result, error_msg)
    metadata = _build_metadata(
        instance_id=instance_id,
        tasks_dir=tasks_dir,
        task_dir=task_dir,
        config_path=config_path,
        started=started,
        result=result,
        reward_payload=reward_payload,
        error_msg=error_msg,
    )

    _write_artifacts(
        output_root=output_root,
        result=result,
        reward_payload=reward_payload,
        metadata=metadata,
    )
    return _build_response(
        reward_payload=reward_payload,
        result=result,
        error_msg=error_msg,
        metadata=metadata,
    )
