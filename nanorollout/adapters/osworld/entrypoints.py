"""OSWorld runner entrypoints that bind task lifecycle to an agent."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from nanorollout.runner import TaskRunRequest, TaskSpec, run_task

from .adapter import OSWorldTaskAdapter

AGENT_REGISTRY = {
    "qwen3vl": "nanorollout.harness.agents.osworld.mm_agents.qwen3vl.Qwen3VLAgent",
    "qwen3-vl": "nanorollout.harness.agents.osworld.mm_agents.qwen3vl.Qwen3VLAgent",
    "qwen3vl-mmagents": "nanorollout.harness.agents.osworld.mm_agents.qwen3vl.Qwen3VLAgent",
    "qwen3-vl-mmagents": "nanorollout.harness.agents.osworld.mm_agents.qwen3vl.Qwen3VLAgent",
}


def _create_agent(agent_name: str, **kwargs: Any) -> Any:
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {agent_name}. Available: {list(AGENT_REGISTRY.keys())}"
        )
    module_path, class_name = AGENT_REGISTRY[agent_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kwargs)


def _build_osworld_agent(task: TaskSpec, request: TaskRunRequest) -> Any:
    params = task.metadata["sampling_params"]
    return _create_agent(
        task.metadata["agent_name"],
        model=request.model_name,
        max_tokens=params.get("max_tokens", 4096),
        top_p=params.get("top_p", 0.9),
        temperature=params.get("temperature", 0.0),
        action_space="pyautogui",
        observation_type=task.environment["observation_type"],
        history_n=request.extra_args.get("history_n", 4),
        coordinate_type=request.extra_args.get("coordinate_type", "relative"),
        base_url=request.base_url,
        api_key=request.api_key,
    )


def run_osworld(
    instance_id: str,
    output_dir: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    env_type: str = "aws",
    sampling_params: Optional[object] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request = TaskRunRequest(
        instance_id=instance_id,
        output_dir=output_dir,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        env_type=env_type,
        sampling_params=sampling_params,
        extra_args=dict(extra_args or {}),
    )
    return run_task(request, OSWorldTaskAdapter(agent_builder=_build_osworld_agent))
