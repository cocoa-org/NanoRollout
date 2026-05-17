"""Terminal runner entrypoints that bind task lifecycle to an agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nanorollout.harness.agents.shared.installed_agent_factory import (
    build_installed_agent,
)
from nanorollout.runner import (
    TaskRunRequest,
    TaskSpec,
)
from nanorollout.harness.agents.shared.llm_config import build_llm_config
from nanorollout.harness.agents.swe.base import AgentConfig as SweAgentConfig

from .adapter import (
    ExitStatusBuilder,
    TerminalAgentBuilder,
    TerminalAgentRunner,
    TerminalTaskSpec,
    default_terminal_exit_status,
    installed_terminal_exit_status,
    run_terminal_agent,
)

RUNNER_RESERVED_ARGS = {
    "agent",
    "agent_kwargs",
    "agent_env",
    "repo_dir",
    "repo_url",
    "repo_revision",
    "refresh_repo",
    "env_timeout",
    "create_timeout",
    "step_timeout",
    "eval_timeout",
    "max_iterations",
}

RunCallable = Callable[..., Dict[str, Any]]
ToolsJsonBuilder = Callable[[Any], Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class TerminalAgentSpec:
    entrypoint: str
    runner_label: str
    agent_builder: TerminalAgentBuilder
    agent_runner: TerminalAgentRunner
    get_tools_json: Optional[ToolsJsonBuilder] = None
    build_exit_status: ExitStatusBuilder = default_terminal_exit_status
    use_sampling_params: bool = True


def _build_miniswe_agent(
    env_obj: Any,
    task: TaskSpec,
    request: TaskRunRequest,
    trial_dir: Path,
) -> Any:
    del trial_dir
    from nanorollout.harness.agents.terminal.miniswe import TerminalMiniSweAgent

    spec: TerminalTaskSpec = task.metadata["terminal_spec"]
    llm_config = build_llm_config(
        model=request.model_name,
        base_url=request.base_url,
        api_key=request.api_key or "abc-123",
        sampling_params=request.sampling_params,
        default_temperature=0.6,
        default_top_p=0.95,
        default_max_tokens=4096,
    )
    agent_config = SweAgentConfig(
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
    return TerminalMiniSweAgent(
        environment=env_obj,
        config=agent_config,
        step_timeout=task.environment.get("env_step_timeout"),
    )


def _run_miniswe_agent(agent: Any, task: TaskSpec, env_obj: Any) -> Any:
    del env_obj
    return agent.run(task.instruction)


def _build_terminus2_agent(
    env_obj: Any,
    task: TaskSpec,
    request: TaskRunRequest,
    trial_dir: Path,
) -> Any:
    from nanorollout.harness.agents.terminal.terminus2 import Terminus2Agent

    spec: TerminalTaskSpec = task.metadata["terminal_spec"]
    llm_config = build_llm_config(
        model=request.model_name,
        base_url=request.base_url,
        api_key=request.api_key or "abc-123",
        sampling_params=request.sampling_params,
        default_temperature=0.6,
    )
    return Terminus2Agent(
        environment=env_obj,
        model=llm_config.model,
        parser_name=request.extra_args.get("parser_name", "json"),
        api_key=llm_config.api_key,
        api_base=llm_config.api_base,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        max_tokens=llm_config.max_tokens,
        max_iterations=spec.max_iterations,
        extra_body=llm_config.extra_body,
        enable_asciinema=request.extra_args.get("enable_asciinema", False),
        output_dir=str(trial_dir),
        enable_summarize=request.extra_args.get("enable_summarize", True),
        proactive_summarization_threshold=request.extra_args.get(
            "proactive_summarization_threshold",
            8000,
        ),
        ubuntu_mirror=request.extra_args.get("ubuntu_mirror", "us.archive.ubuntu.com"),
        ubuntu_mirror_apt_update=request.extra_args.get(
            "ubuntu_mirror_apt_update",
            True,
        ),
    )


def _run_terminus2_agent(agent: Any, task: TaskSpec, env_obj: Any) -> Any:
    del env_obj
    return agent.run(
        task.instruction, total_timeout=task.environment.get("agent_timeout")
    )


def _installed_agent_builder(agent_name: str) -> TerminalAgentBuilder:
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
    task: TaskSpec,
    env_obj: Any,
) -> Any:
    return agent.run(
        task.instruction,
        env_obj,
        timeout_sec=int(task.environment["agent_timeout"])
        if task.environment.get("agent_timeout")
        else None,
    )


def _no_tools(agent: Any) -> None:
    del agent
    return None


def _make_terminal_runner(spec: TerminalAgentSpec) -> RunCallable:
    def run(
        instance_id: str,
        output_dir: str,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        env_type: str = "docker",
        sampling_params: Optional[object] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return run_terminal_agent(
            instance_id=instance_id,
            output_dir=output_dir,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            env_type=env_type,
            sampling_params=sampling_params if spec.use_sampling_params else None,
            extra_args=extra_args,
            runner_label=spec.runner_label,
            agent_builder=spec.agent_builder,
            agent_runner=spec.agent_runner,
            get_tools_json=spec.get_tools_json,
            build_exit_status=spec.build_exit_status,
        )

    run.__name__ = spec.entrypoint
    run.__qualname__ = spec.entrypoint
    run.__doc__ = f"Run {spec.runner_label} on Terminal Bench tasks."
    return run


def _installed_spec(agent_name: str, entrypoint: str) -> TerminalAgentSpec:
    return TerminalAgentSpec(
        entrypoint=entrypoint,
        runner_label=f"installed agent {agent_name}",
        agent_builder=_installed_agent_builder(agent_name),
        agent_runner=_run_installed_agent_result,
        get_tools_json=_no_tools,
        build_exit_status=installed_terminal_exit_status,
        use_sampling_params=False,
    )


run_tb_miniswe = _make_terminal_runner(
    TerminalAgentSpec(
        entrypoint="run_tb_miniswe",
        runner_label="TerminalMiniSweAgent",
        agent_builder=_build_miniswe_agent,
        agent_runner=_run_miniswe_agent,
    )
)

run_tb_terminus2 = _make_terminal_runner(
    TerminalAgentSpec(
        entrypoint="run_tb_terminus2",
        runner_label="Terminus-2 agent",
        agent_builder=_build_terminus2_agent,
        agent_runner=_run_terminus2_agent,
        get_tools_json=_no_tools,
    )
)

run_tb_claude_code = _make_terminal_runner(
    _installed_spec("claude-code", "run_tb_claude_code")
)
run_tb_qwen_code = _make_terminal_runner(
    _installed_spec("qwen-code", "run_tb_qwen_code")
)
run_tb_opencode = _make_terminal_runner(_installed_spec("opencode", "run_tb_opencode"))
