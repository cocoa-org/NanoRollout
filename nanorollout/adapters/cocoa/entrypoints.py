"""Cocoa runner entrypoints that bind task lifecycle to a runtime."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from nanorollout.runner import TaskRunRequest, run_task

from .adapter import CocoaTaskAdapter


def _ensure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        root.setLevel(level)


def _build_cocoa_runtime(config: dict[str, Any], log_level: int) -> Any:
    del log_level
    from nanorollout.envs.cocoa_env import setup_logging
    from nanorollout.harness.agents.cocoa import CocoaAgent

    setup_logging(str(config.get("log_level", "INFO")))
    return CocoaAgent(config)


def run_cocoa_agent(
    instance_id: str,
    output_dir: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    env_type: str = "docker",
    sampling_params: Optional[object] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra_args = dict(extra_args or {})
    log_level_name = str(extra_args.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    _ensure_logging(log_level)
    request = TaskRunRequest(
        instance_id=instance_id,
        output_dir=output_dir,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        env_type=env_type or "docker",
        sampling_params=sampling_params,
        extra_args=extra_args,
    )
    return run_task(
        request,
        CocoaTaskAdapter(runtime_builder=_build_cocoa_runtime),
    )
