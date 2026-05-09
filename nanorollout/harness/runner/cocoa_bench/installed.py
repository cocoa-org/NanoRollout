"""Cocoa-Bench runner entrypoints for Harbor-style installed agents."""

from __future__ import annotations

import inspect
import logging
import time
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, Dict, Optional

from nanorollout.harness.agents.shared import ClaudeCode, OpenCode, QwenCode
from nanorollout.harness.runner.cocoa_bench.common import (
    _attach_trial_log,
    _allocate_port,
    _build_cocoa_config,
    _build_installed_agent_prompt,
    _build_installed_result,
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
)

logger = logging.getLogger(__name__)

AGENT_REGISTRY = {
    "claude-code": ClaudeCode,
    "qwen-code": QwenCode,
    "qwen-coder": QwenCode,
    "opencode": OpenCode,
}

RUNNER_RESERVED_ARGS = {
    "agent_kwargs",
    "agent_env",
    "repo_dir",
    "repo_url",
    "repo_revision",
    "refresh_repo",
    "tasks_dir",
    "tasks_subdir",
    "config_path",
    "controller_type",
    "controller_args",
    "sandbox_config",
    "client_type",
    "runtime_type",
    "agent_type",
    "use_encrypted_tasks",
    "create_timeout",
    "env_timeout",
    "step_timeout",
    "eval_timeout",
    "max_iterations",
    "docker_port",
    "browser_resolution",
    "modal_app_name",
    "modal_timeout",
    "modal_idle_timeout",
    "modal_startup_timeout",
    "modal_container_port",
    "workspace_dir",
    "log_level",
}

PROVIDER_API_ENV = {
    "amazon-bedrock": "AWS_ACCESS_KEY_ID",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "github-copilot": "GITHUB_TOKEN",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "HF_TOKEN",
    "llama": "LLAMA_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}


def _inject_agent_credentials(
    agent_name: str,
    model_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    extra_env: dict[str, str],
) -> None:
    if agent_name == "claude-code":
        if api_key:
            extra_env.setdefault("ANTHROPIC_API_KEY", api_key)
        if base_url:
            extra_env.setdefault("ANTHROPIC_BASE_URL", base_url)
        return

    if agent_name == "qwen-code":
        if api_key:
            extra_env.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            extra_env.setdefault("OPENAI_BASE_URL", base_url)
        return

    if agent_name == "opencode":
        provider = model_name.split("/", 1)[0] if "/" in model_name else "openai"
        env_key = PROVIDER_API_ENV.get(provider)
        if api_key and env_key:
            extra_env.setdefault(env_key, api_key)
        if base_url and provider == "openai":
            extra_env.setdefault("OPENAI_BASE_URL", base_url)


def _build_agent(
    agent_name: str,
    *,
    trial_dir: Path,
    model_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    extra_args: Dict[str, Any],
):
    agent_cls = AGENT_REGISTRY[agent_name]
    agent_kwargs = dict(extra_args.get("agent_kwargs") or {})
    for key, value in extra_args.items():
        if key in RUNNER_RESERVED_ARGS:
            continue
        agent_kwargs.setdefault(key, value)

    extra_env = dict(extra_args.get("agent_env", {}) or {})
    extra_env.update(agent_kwargs.pop("extra_env", {}) or {})
    extra_env.update(agent_kwargs.pop("agent_env", {}) or {})
    _inject_agent_credentials(agent_name, model_name, api_key, base_url, extra_env)

    agent_kwargs["logs_dir"] = trial_dir / "agent"
    agent_kwargs["model_name"] = model_name
    agent_kwargs["extra_env"] = extra_env
    return agent_cls(**agent_kwargs)


def _run_installed_agent(
    agent_name: str,
    instance_id: str,
    output_dir: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    env_type: str = "docker",
    sampling_params: Optional[object] = None,
    extra_args: Dict[str, Any] = {},
) -> Dict[str, Any]:
    extra_args = dict(extra_args or {})
    env_type = env_type or "docker"
    sampling_params_dict = _parse_sampling_params(sampling_params)
    log_level_name = str(extra_args.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    _ensure_logging(log_level)
    started = time.time()

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trial_log_path = output_root / "trial.log"
    config_path = output_root / "cocoa_config.json"
    result: dict[str, Any] = {}
    error_msg: Optional[str] = None
    tasks_dir = output_root
    task_dir = output_root

    with _attach_trial_log(trial_log_path, log_level):
        logger.info("[%s] Writing Cocoa installed-agent log to %s", instance_id, trial_log_path)
        try:
            tasks_dir, task_dir = _resolve_task_root(instance_id, extra_args)
            encrypted_task = _detect_encrypted_task(task_dir)
            preferred_port = extra_args.get("docker_port")
            docker_port = _allocate_port(int(preferred_port) if preferred_port is not None else None)
            config_args = dict(extra_args)
            config_args.setdefault("agent_type", agent_name)
            config = _build_cocoa_config(
                model_name=model_name,
                base_url=None,
                api_key=None,
                env_type=env_type,
                sampling_params=sampling_params_dict,
                extra_args=config_args,
                encrypted_task=encrypted_task,
                docker_port=docker_port,
            )
            _write_json(config_path, config)

            from nanorollout.envs.cocoa_env import TaskExecutor, setup_logging
            from nanorollout.harness.agents.cocoa.controller import Human

            setup_logging(
                str(config.get("log_level", log_level_name)),
                log_file=str(trial_log_path),
            )

            task = _load_task(
                task_dir,
                _coerce_bool(config.get("use_encrypted_tasks"), default=encrypted_task),
            )
            executor = TaskExecutor(config, controller=Human())
            wait_time = int(
                extra_args.get("create_timeout")
                or extra_args.get("env_timeout")
                or 30
            )
            timeout_value = int(
                extra_args.get("step_timeout")
                or extra_args.get("env_timeout")
                or extra_args.get("create_timeout")
                or 1800
            )

            logger.info("[%s] Running %s Cocoa task from %s", instance_id, agent_name, task_dir)
            try:
                executor.setup_environment(task, wait_time=wait_time)
                if not hasattr(executor.sandbox_client, "execute"):
                    raise TypeError(
                        "Cocoa installed agents require a unified sandbox client with terminal support"
                    )
                executor.sandbox_client.timeout = timeout_value
                agent = _build_agent(
                    agent_name,
                    trial_dir=output_root,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    extra_args=extra_args,
                )
                agent_start = time.time()
                agent_result = agent.run(
                    _build_installed_agent_prompt(task),
                    executor.sandbox_client,
                    timeout_sec=timeout_value,
                )
                result = _build_installed_result(
                    agent_name=agent_name,
                    model_name=model_name,
                    agent_result=agent_result,
                    sandbox_runtime=executor.sandbox_client.get_runtime_metadata(),
                    agent_time=time.time() - agent_start,
                )
                eval_result = executor.run_eval(task, result)
                if eval_result is not None:
                    result["eval"] = eval_result
            finally:
                try:
                    executor.cleanup_environment()
                except Exception as cleanup_exc:
                    logger.exception("Installed Cocoa cleanup failed for %s", instance_id)
                    if error_msg is None:
                        error_msg = f"Cleanup failed: {cleanup_exc}"
        except Exception as exc:
            error_msg = str(exc)
            logger.exception("Installed Cocoa run failed for %s", instance_id)

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


def _make_installed_runner(agent_name: str, *, default_env_type: str):
    runner = partial(_run_installed_agent, agent_name, env_type=default_env_type)
    update_wrapper(runner, _run_installed_agent)
    runner.__name__ = f"run_cocoa_{agent_name.replace('-', '_')}"
    runner.__qualname__ = runner.__name__
    signature = inspect.signature(_run_installed_agent)
    parameters = []
    for parameter in signature.parameters.values():
        if parameter.name == "agent_name":
            continue
        if parameter.name == "env_type":
            parameter = parameter.replace(default=default_env_type)
        parameters.append(parameter)
    runner.__signature__ = signature.replace(parameters=parameters)
    runner.__doc__ = (
        f"Run the {agent_name} installed agent for Cocoa-Bench "
        f"(default env_type={default_env_type!r})."
    )
    return runner


run_cocoa_claude_code = _make_installed_runner(
    "claude-code",
    default_env_type="docker",
)
run_cocoa_qwen_code = _make_installed_runner(
    "qwen-code",
    default_env_type="docker",
)
run_cocoa_opencode = _make_installed_runner(
    "opencode",
    default_env_type="docker",
)
