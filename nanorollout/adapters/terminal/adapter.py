"""Terminal Bench adapter for the common task lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nanorollout.runner import (
    TaskAdapter,
    TaskRunRequest,
    TaskSpec,
    require_args,
    resolve_completed_status,
    run_task,
)

from .common import (
    DEFAULT_TERMINAL_BENCH_REPO_URL,
    ENV_LOGGER_NAME,
    build_reward_payload,
    create_environment,
    ensure_tb_image,
    resolve_tb_instance,
    resolve_tb_repo_dir,
    resolve_tb_timeouts,
)

TerminalAgentBuilder = Callable[[Any, TaskSpec, TaskRunRequest, Path], Any]
TerminalAgentRunner = Callable[[Any, TaskSpec, Any], Any]
ExitStatusBuilder = Callable[[Optional[str], Any, Dict[str, Any]], str]


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass(frozen=True)
class TerminalTaskSpec:
    env_timeout: int
    create_timeout: int
    max_iterations: int
    repo_dir: Optional[str] = None
    repo_url: str = DEFAULT_TERMINAL_BENCH_REPO_URL
    repo_revision: Optional[str] = None
    refresh_repo: bool = False
    step_timeout: Optional[int] = None
    eval_timeout: Optional[int] = None
    timeout_multiplier: float = 1.0

    @classmethod
    def from_request(cls, request: TaskRunRequest) -> "TerminalTaskSpec":
        args = request.extra_args
        require_args(
            args, ("env_timeout", "create_timeout", "max_iterations"), "Terminal task"
        )
        return cls(
            env_timeout=int(args["env_timeout"]),
            create_timeout=int(args["create_timeout"]),
            max_iterations=int(args["max_iterations"]),
            repo_dir=args.get("repo_dir"),
            repo_url=str(args.get("repo_url", DEFAULT_TERMINAL_BENCH_REPO_URL)),
            repo_revision=args.get("repo_revision"),
            refresh_repo=_coerce_bool(args.get("refresh_repo", False)),
            step_timeout=_optional_int(args.get("step_timeout")),
            eval_timeout=_optional_int(args.get("eval_timeout")),
            timeout_multiplier=float(args.get("timeout_multiplier", 1.0)),
        )

    def timeout_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {"timeout_multiplier": self.timeout_multiplier}
        if self.step_timeout is not None:
            args["step_timeout"] = self.step_timeout
        if self.eval_timeout is not None:
            args["eval_timeout"] = self.eval_timeout
        return args


def _evaluate_terminal(
    env_obj: Any,
    task: TaskSpec,
    request: TaskRunRequest,
    trial_dir: Path,
) -> tuple[Dict[str, Any], Optional[str]]:
    del trial_dir
    from nanorollout.adapters.terminal.task import run_tb_eval

    tests_dir = str(Path(task.evaluation["repo_dir"]) / request.instance_id / "tests")
    return run_tb_eval(
        env_obj=env_obj,
        instance=task.payload,
        eval_timeout=task.evaluation.get("eval_timeout"),
        tests_dir=tests_dir,
    )


def default_terminal_exit_status(
    error_msg: Optional[str],
    agent_result: Any,
    eval_payload: Dict[str, Any],
) -> str:
    del agent_result
    return "Error" if error_msg else resolve_completed_status(eval_payload)


def installed_terminal_exit_status(
    error_msg: Optional[str],
    agent_result: Any,
    eval_payload: Dict[str, Any],
) -> str:
    if error_msg:
        return "Error"
    if agent_result and getattr(agent_result, "error", None):
        return getattr(agent_result, "exit_status", "Error")
    return resolve_completed_status(eval_payload)


@dataclass
class TerminalTaskAdapter(TaskAdapter):
    runner_label: str
    agent_builder: TerminalAgentBuilder
    agent_runner: TerminalAgentRunner
    tools_json_builder: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None
    exit_status_builder: ExitStatusBuilder = default_terminal_exit_status
    env_logger_name: str = ENV_LOGGER_NAME
    eval_logger_name: str = "nanorollout.adapters.terminal.task.grading"

    def create_environment(
        self,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> Any:
        spec = task.metadata["terminal_spec"]
        return create_environment(
            env_type=request.env_type,
            instance=task.payload,
            image=task.environment.get("image", ""),
            workspace_dir=task.environment.get("workspace_dir", "/"),
            env_timeout=spec.env_timeout,
            create_timeout=spec.create_timeout,
            step_timeout=task.environment.get("env_step_timeout"),
            eval_timeout=task.environment.get("env_eval_timeout"),
        )

    def describe_environment(
        self,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> str:
        image = task.environment.get("image")
        return f"{request.env_type} {image}" if image else request.env_type

    def prepare_task(
        self,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> TaskSpec:
        del trial_dir
        spec = TerminalTaskSpec.from_request(request)
        repo_dir = resolve_tb_repo_dir(
            spec.repo_dir,
            repo_url=spec.repo_url,
            revision=spec.repo_revision,
            refresh=spec.refresh_repo,
        )
        instance = resolve_tb_instance(
            instance_id=request.instance_id, repo_dir=repo_dir
        )
        image = ensure_tb_image(instance, env_type=request.env_type)
        agent_timeout, eval_timeout = resolve_tb_timeouts(instance, spec.timeout_args())
        return TaskSpec(
            id=request.instance_id,
            kind="terminal",
            payload=instance,
            instruction=instance["instruction"],
            environment={
                "image": image,
                "workspace_dir": instance.get("workspace_dir") or "/",
                "agent_timeout": agent_timeout,
                "env_step_timeout": spec.step_timeout or agent_timeout,
                "env_eval_timeout": spec.eval_timeout or eval_timeout,
            },
            evaluation={
                "repo_dir": repo_dir,
                "eval_timeout": eval_timeout,
            },
            metadata={
                "terminal_spec": spec,
            },
        )

    def build_agent(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> Any:
        return self.agent_builder(env_obj, task, request, trial_dir)

    def run_agent(
        self,
        agent: Any,
        task: TaskSpec,
        env_obj: Any,
    ) -> Any:
        return self.agent_runner(agent, task, env_obj)

    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        return _evaluate_terminal(env_obj, task, request, trial_dir)

    def build_reward_payload(
        self,
        instance_id: str,
        eval_payload: Dict[str, Any],
        error_msg: Optional[str],
    ) -> Dict[str, Any]:
        return build_reward_payload(instance_id, eval_payload, error_msg)

    def get_tools_json(self, agent: Any) -> Optional[Dict[str, Any]]:
        if self.tools_json_builder:
            return self.tools_json_builder(agent)
        return super().get_tools_json(agent)

    def build_exit_status(
        self,
        error_msg: Optional[str],
        agent_result: Any,
        eval_payload: Dict[str, Any],
    ) -> str:
        return self.exit_status_builder(error_msg, agent_result, eval_payload)


def run_terminal_agent(
    *,
    instance_id: str,
    output_dir: str,
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    env_type: str,
    sampling_params: Optional[object],
    extra_args: Optional[Dict[str, Any]],
    runner_label: str,
    agent_builder: TerminalAgentBuilder,
    agent_runner: TerminalAgentRunner,
    get_tools_json: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None,
    build_exit_status: ExitStatusBuilder = default_terminal_exit_status,
) -> Dict[str, Any]:
    extra_args = dict(extra_args or {})

    request = TaskRunRequest(
        instance_id=instance_id,
        output_dir=output_dir,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        env_type=env_type,
        sampling_params=sampling_params,
        extra_args=extra_args,
    )
    adapter = TerminalTaskAdapter(
        runner_label=runner_label,
        agent_builder=agent_builder,
        agent_runner=agent_runner,
        tools_json_builder=get_tools_json,
        exit_status_builder=build_exit_status,
    )
    return run_task(request, adapter)
