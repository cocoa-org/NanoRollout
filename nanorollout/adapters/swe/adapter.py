"""SWE benchmark adapter for the common task lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nanorollout.runner import (
    TaskAdapter,
    TaskRunRequest,
    TaskSpec,
    require_args,
    run_task,
)

from .common import (
    ENV_LOGGER_NAME,
    build_reward_payload,
    create_environment,
    write_artifacts,
)
from .task import resolve_swe_dataset_adapter

TaskBuilder = Callable[[Dict[str, Any], str, Optional[str]], str]
AgentBuilder = Callable[[Any, TaskSpec, TaskRunRequest, Path], Any]
AgentRunner = Callable[[Any, str, int, Any], Any]
ResultHook = Callable[[Any], None]


@dataclass(frozen=True)
class SweTaskSpec:
    dataset: str
    split: str
    step_timeout: int
    eval_timeout: int
    env_timeout: int
    create_timeout: int
    max_iterations: int

    @classmethod
    def from_request(cls, request: TaskRunRequest) -> "SweTaskSpec":
        args = request.extra_args
        require_args(
            args,
            (
                "dataset",
                "split",
                "step_timeout",
                "eval_timeout",
                "env_timeout",
                "create_timeout",
                "max_iterations",
            ),
            "SWE task",
        )
        return cls(
            dataset=str(args["dataset"]),
            split=str(args["split"]),
            step_timeout=int(args["step_timeout"]),
            eval_timeout=int(args["eval_timeout"]),
            env_timeout=int(args["env_timeout"]),
            create_timeout=int(args["create_timeout"]),
            max_iterations=int(args["max_iterations"]),
        )


def _default_run_agent(agent: Any, task: str, step_timeout: int, env_obj: Any) -> Any:
    del step_timeout, env_obj
    return agent.run(task)


def _swe_exit_status(
    error_msg: Optional[str],
    agent_result: Any,
    eval_payload: Dict[str, Any],
) -> str:
    del eval_payload
    if error_msg:
        return "error"
    return getattr(agent_result, "exit_status", "error")


@dataclass
class SweTaskAdapter(TaskAdapter):
    runner_label: str
    task_builder: TaskBuilder
    agent_builder: AgentBuilder
    agent_runner: Optional[AgentRunner] = None
    result_hook: Optional[ResultHook] = None
    env_logger_name: str = ENV_LOGGER_NAME
    eval_logger_name: str = "nanorollout.adapters.swe.task"

    def create_environment(
        self,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> Any:
        spec = task.metadata["swe_spec"]
        return create_environment(
            env_type=request.env_type,
            instance=task.payload,
            image=task.environment.get("image", ""),
            workspace_dir=task.environment.get("workspace_dir", "/"),
            env_timeout=spec.env_timeout,
            create_timeout=spec.create_timeout,
            step_timeout=spec.step_timeout,
            eval_timeout=spec.eval_timeout,
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
        spec = SweTaskSpec.from_request(request)
        dataset = resolve_swe_dataset_adapter(spec.dataset)
        instance = dataset.resolve_instance(request.instance_id, spec.split)
        workspace_dir = dataset.workspace_dir()
        base_commit = instance.get("base_commit")
        return TaskSpec(
            id=request.instance_id,
            kind="swe",
            payload=instance,
            instruction=self.task_builder(instance, workspace_dir, base_commit),
            environment={
                "image": dataset.image_name(instance, request.env_type, request),
                "workspace_dir": workspace_dir,
                "agent_timeout": spec.step_timeout,
                "env_step_timeout": spec.step_timeout,
                "env_eval_timeout": spec.eval_timeout,
            },
            evaluation={
                "dataset_adapter": dataset,
                "eval_timeout": spec.eval_timeout,
            },
            metadata={
                "swe_spec": spec,
                "base_commit": base_commit,
            },
        )

    def setup_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        env_obj.execute("git config --global core.pager ''")
        env_obj.execute("git config --global diff.binary false")
        task.evaluation["dataset_adapter"].setup_environment(env_obj, task, request)
        if task.metadata.get("base_commit"):
            env_obj._base_commit = task.metadata["base_commit"]

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
        runner = self.agent_runner or _default_run_agent
        return runner(
            agent,
            task.instruction,
            int(task.environment.get("agent_timeout") or 0),
            env_obj,
        )

    def after_agent_result(self, agent_result: Any) -> None:
        if self.result_hook:
            self.result_hook(agent_result)

    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        del trial_dir
        return task.evaluation["dataset_adapter"].evaluate(env_obj, task, request)

    def build_reward_payload(
        self,
        instance_id: str,
        eval_payload: Dict[str, Any],
        error_msg: Optional[str],
    ) -> Dict[str, Any]:
        return build_reward_payload(instance_id, eval_payload, error_msg)

    def write_result(
        self,
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
        write_artifacts(
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

    def build_exit_status(
        self,
        error_msg: Optional[str],
        agent_result: Any,
        eval_payload: Dict[str, Any],
    ) -> str:
        return _swe_exit_status(error_msg, agent_result, eval_payload)


def run_swe_agent(
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
    task_builder: TaskBuilder,
    agent_builder: AgentBuilder,
    agent_runner: Optional[AgentRunner] = None,
    result_hook: Optional[ResultHook] = None,
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
    adapter = SweTaskAdapter(
        runner_label=runner_label,
        task_builder=task_builder,
        agent_builder=agent_builder,
        agent_runner=agent_runner,
        result_hook=result_hook,
    )
    return run_task(request, adapter)
