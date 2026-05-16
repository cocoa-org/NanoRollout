"""SWE runner entrypoints that bind task lifecycle to an agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nanorollout.harness.agents.swe.openhands.prompts import (
    build_core_user_prompt,
    build_lite_user_prompt,
)
from nanorollout.harness.agents.shared.installed_agent_factory import (
    build_installed_agent,
)
from nanorollout.harness.agents.shared.llm_config import build_llm_config
from nanorollout.harness.agents.swe.base import AgentConfig as SweAgentConfig
from nanorollout.runner import TaskRunRequest, TaskSpec

from .adapter import (
    AgentBuilder,
    AgentRunner,
    ResultHook,
    SweTaskSpec,
    TaskBuilder,
    run_swe_agent,
)

RUNNER_RESERVED_ARGS = {
    "dataset",
    "split",
    "step_timeout",
    "eval_timeout",
    "env_timeout",
    "create_timeout",
    "max_iterations",
    "agent_kwargs",
    "agent_env",
    "swebench_pro_scripts_dir",
    "swebench_pro_repo",
    "swebench_pro_dockerhub_username",
    "pro_scripts_dir",
    "pro_repo",
    "dockerhub_username",
}

RunCallable = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class SweAgentSpec:
    entrypoint: str
    runner_label: str
    task_builder: TaskBuilder
    agent_builder: AgentBuilder
    agent_runner: Optional[AgentRunner] = None
    result_hook: Optional[ResultHook] = None
    use_sampling_params: bool = True


def _raw_problem_statement(
    instance: Dict[str, Any],
    workspace_dir: str,
    base_commit: Optional[str],
) -> str:
    del workspace_dir, base_commit
    return instance["problem_statement"]


def _core_prompt(
    instance: Dict[str, Any],
    workspace_dir: str,
    base_commit: Optional[str],
) -> str:
    return build_core_user_prompt(
        workspace_dir=workspace_dir,
        problem_statement=instance["problem_statement"],
        base_commit=base_commit,
    )


def _lite_prompt(
    instance: Dict[str, Any],
    workspace_dir: str,
    base_commit: Optional[str],
) -> str:
    return build_lite_user_prompt(
        workspace_dir=workspace_dir,
        problem_statement=instance["problem_statement"],
        base_commit=base_commit,
    )


def _build_agent_config(
    task: TaskSpec,
    request: TaskRunRequest,
) -> SweAgentConfig:
    spec: SweTaskSpec = task.metadata["swe_spec"]
    llm_config = build_llm_config(
        model=request.model_name,
        base_url=request.base_url,
        api_key=request.api_key or "abc-123",
        sampling_params=request.sampling_params,
        default_temperature=0.6,
        default_top_p=0.95,
        default_max_tokens=4096,
    )
    return SweAgentConfig(
        model=llm_config.model,
        max_iterations=spec.max_iterations,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p if llm_config.top_p is not None else 0.95,
        max_tokens=llm_config.max_tokens or 4096,
        extra_body=llm_config.extra_body,
        api_key=llm_config.api_key,
        api_base=llm_config.api_base,
        llm_provider=llm_config.llm_provider,
    )


def _build_miniswe_agent(
    env_obj: Any,
    task: TaskSpec,
    request: TaskRunRequest,
    trial_dir: Path,
) -> Any:
    del trial_dir
    from nanorollout.harness.agents.swe.mini_swe_agent import MiniSweAgent

    return MiniSweAgent(
        environment=env_obj,
        config=_build_agent_config(task, request),
        step_timeout=int(task.environment.get("agent_timeout") or 0),
    )


def _build_oh_core_agent(
    env_obj: Any,
    task: TaskSpec,
    request: TaskRunRequest,
    trial_dir: Path,
) -> Any:
    del trial_dir
    from nanorollout.harness.agents.swe.openhands.codeact import CodeActAgent

    return CodeActAgent(environment=env_obj, config=_build_agent_config(task, request))


def _build_oh_lite_agent(
    env_obj: Any,
    task: TaskSpec,
    request: TaskRunRequest,
    trial_dir: Path,
) -> Any:
    del trial_dir
    from nanorollout.harness.agents.swe.openhands.codeact_lite import CodeActLiteAgent

    return CodeActLiteAgent(
        environment=env_obj, config=_build_agent_config(task, request)
    )


def _build_r2egym_agent(
    env_obj: Any,
    task: TaskSpec,
    request: TaskRunRequest,
    trial_dir: Path,
) -> Any:
    del trial_dir
    from nanorollout.harness.agents.swe.r2egym import R2EGymAgent

    return R2EGymAgent(
        environment=env_obj,
        config=_build_agent_config(task, request),
        use_fn_calling=request.extra_args.get("use_fn_calling", True),
    )


def _submitted_output_as_patch(agent_result: Any) -> None:
    if agent_result is None:
        return
    submitted_patch = agent_result.message if agent_result.success else ""
    agent_result.patch = submitted_patch or ""


def _installed_agent_builder(agent_name: str) -> AgentBuilder:
    def build_agent(
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> Any:
        del env_obj, task
        return build_installed_agent(
            agent_name,
            trial_dir=trial_dir,
            model_name=request.model_name,
            api_key=request.api_key,
            base_url=request.base_url,
            extra_args=request.extra_args,
            reserved_args=RUNNER_RESERVED_ARGS,
        )

    return build_agent


def _run_installed_agent_result(
    agent: Any,
    task: str,
    step_timeout: int,
    env_obj: Any,
) -> Any:
    return agent.run(task, env_obj, timeout_sec=step_timeout)


def _make_swe_runner(spec: SweAgentSpec) -> RunCallable:
    def run(
        instance_id: str,
        output_dir: str,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        env_type: str = "enroot",
        sampling_params: Optional[object] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return run_swe_agent(
            instance_id=instance_id,
            output_dir=output_dir,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            env_type=env_type,
            sampling_params=sampling_params if spec.use_sampling_params else None,
            extra_args=extra_args,
            runner_label=spec.runner_label,
            task_builder=spec.task_builder,
            agent_builder=spec.agent_builder,
            agent_runner=spec.agent_runner,
            result_hook=spec.result_hook,
        )

    run.__name__ = spec.entrypoint
    run.__qualname__ = spec.entrypoint
    run.__doc__ = f"Run {spec.runner_label} on SWE-Bench style tasks."
    return run


def _installed_spec(agent_name: str, entrypoint: str) -> SweAgentSpec:
    return SweAgentSpec(
        entrypoint=entrypoint,
        runner_label=f"installed agent {agent_name}",
        task_builder=_raw_problem_statement,
        agent_builder=_installed_agent_builder(agent_name),
        agent_runner=_run_installed_agent_result,
        use_sampling_params=False,
    )


run_miniswe = _make_swe_runner(
    SweAgentSpec(
        entrypoint="run_miniswe",
        runner_label="MiniSweAgent",
        task_builder=_raw_problem_statement,
        agent_builder=_build_miniswe_agent,
        result_hook=_submitted_output_as_patch,
    )
)

run_oh_core = _make_swe_runner(
    SweAgentSpec(
        entrypoint="run_oh_core",
        runner_label="CodeActAgent",
        task_builder=_core_prompt,
        agent_builder=_build_oh_core_agent,
    )
)

run_oh_lite = _make_swe_runner(
    SweAgentSpec(
        entrypoint="run_oh_lite",
        runner_label="CodeActLiteAgent",
        task_builder=_lite_prompt,
        agent_builder=_build_oh_lite_agent,
    )
)

run_r2egym = _make_swe_runner(
    SweAgentSpec(
        entrypoint="run_r2egym",
        runner_label="R2EGymAgent",
        task_builder=_raw_problem_statement,
        agent_builder=_build_r2egym_agent,
    )
)

run_installed_claude_code = _make_swe_runner(
    _installed_spec("claude-code", "run_installed_claude_code")
)
run_installed_qwen_code = _make_swe_runner(
    _installed_spec("qwen-code", "run_installed_qwen_code")
)
run_installed_opencode = _make_swe_runner(
    _installed_spec("opencode", "run_installed_opencode")
)
