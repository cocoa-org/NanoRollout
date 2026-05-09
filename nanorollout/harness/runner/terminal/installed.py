"""Terminal Bench runner entrypoints for Harbor-style installed agents."""

from __future__ import annotations

import gc
import inspect
import logging
import time
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, Dict, Optional

from nanorollout.harness.agents.shared import ClaudeCode, OpenCode, QwenCode
from nanorollout.harness.runner.terminal.common import (
    DEFAULT_TERMINAL_BENCH_REPO_URL,
    _build_agent_metrics,
    _build_metadata,
    _ensure_logging,
    _resolve_exit_status,
    _write_artifacts,
    build_reward_payload,
    create_environment,
    env_logging,
    eval_logging,
    ensure_tb_image,
    resolve_tb_instance,
    resolve_tb_repo_dir,
    resolve_tb_timeouts,
    trial_logging,
)

logger = logging.getLogger(__name__)

AGENT_REGISTRY = {
    "claude-code": ClaudeCode,
    "qwen-code": QwenCode,
    "qwen-coder": QwenCode,
    "opencode": OpenCode,
}

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
        if key in RUNNER_RESERVED_ARGS or key == "agent_kwargs":
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


def _run_tb_installed_agent(
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
    del sampling_params

    from nanorollout.eval.terminalbench.grading import run_tb_eval

    for key in ("env_timeout", "create_timeout", "max_iterations"):
        if key not in extra_args:
            raise ValueError(f"Missing required env argument: {key}")

    step_timeout = extra_args.get("step_timeout")
    eval_timeout = extra_args.get("eval_timeout")
    env_timeout = extra_args["env_timeout"]
    create_timeout = extra_args["create_timeout"]

    repo_dir = resolve_tb_repo_dir(
        extra_args.get("repo_dir"),
        repo_url=extra_args.get("repo_url", DEFAULT_TERMINAL_BENCH_REPO_URL),
        revision=extra_args.get("repo_revision"),
        refresh=extra_args.get("refresh_repo", False),
    )

    _ensure_logging()
    started = time.time()
    env_obj = None
    agent_result = None
    eval_payload: Dict[str, Any] = {}
    eval_output: Optional[str] = None
    error_msg: Optional[str] = None
    agent_time = 0.0
    eval_time = 0.0

    trial_dir = Path(output_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        instance = resolve_tb_instance(instance_id=instance_id, repo_dir=repo_dir)
        image_name = ensure_tb_image(instance, env_type=env_type)

        agent_timeout, resolved_eval_timeout = resolve_tb_timeouts(instance, extra_args)
        env_step_timeout = step_timeout or agent_timeout
        env_eval_timeout = eval_timeout or resolved_eval_timeout

        with env_logging(trial_dir):
            with trial_logging(trial_dir):
                logger.info("[%s] Starting %s container %s", instance_id, env_type, image_name)
                workspace_dir = instance.get("workspace_dir") or "/"
                env_obj = create_environment(
                    env_type=env_type,
                    instance=instance,
                    image=image_name,
                    workspace_dir=workspace_dir,
                    env_timeout=env_timeout,
                    create_timeout=create_timeout,
                    step_timeout=env_step_timeout,
                    eval_timeout=env_eval_timeout,
                )
                env_obj.set_tool_log_context(instance_id)
                env_obj.start()

                instruction = instance["instruction"]
                agent = _build_agent(
                    agent_name,
                    trial_dir=trial_dir,
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    extra_args=extra_args,
                )
                logger.info("[%s] Running installed agent %s", instance_id, agent_name)
                agent_start = time.time()
                agent_result = agent.run(
                    instruction,
                    env_obj,
                    timeout_sec=int(agent_timeout) if agent_timeout else None,
                )
                agent_time = time.time() - agent_start

                logger.info("[%s] Running Terminal Bench eval", instance_id)
                eval_start = time.time()
                tests_dir = str(Path(repo_dir) / instance_id / "tests")
                with eval_logging(trial_dir):
                    eval_payload, eval_output = run_tb_eval(
                        env_obj=env_obj,
                        instance=instance,
                        eval_timeout=resolved_eval_timeout,
                        tests_dir=tests_dir,
                    )
                eval_time = time.time() - eval_start
    except Exception as exc:
        error_msg = str(exc)
        logger.exception("Installed agent run failed for %s", instance_id)
    finally:
        if env_obj:
            env_obj.stop()

    reward_payload = build_reward_payload(instance_id, eval_payload, error_msg)
    metadata = _build_metadata(
        instance_id=instance_id,
        env_type=env_type,
        eval_payload=eval_payload,
        error_msg=error_msg,
        trial_dir=trial_dir,
        eval_output=eval_output,
        agent_result=agent_result,
        reward_payload=reward_payload,
    )

    _write_artifacts(
        trial_dir=trial_dir,
        instance_id=instance_id,
        model=model_name,
        base_url=base_url,
        env_type=env_type,
        agent_result=agent_result,
        tools_json=None,
        reward_payload=reward_payload,
        eval_output=eval_output,
        started=started,
        metadata=metadata,
    )

    messages = agent_result.history if agent_result else []
    agent_metrics = _build_agent_metrics(
        messages=messages,
        agent_time=agent_time,
        eval_time=eval_time,
        total_time=time.time() - started,
    )

    exit_status = "Error" if error_msg else (
        agent_result.exit_status if agent_result and agent_result.error else _resolve_exit_status(eval_payload)
    )
    gc.collect()
    return {
        "reward": reward_payload.get("reward", 0),
        "messages": messages,
        "exit_status": exit_status,
        "agent_metrics": agent_metrics,
        "metadata": metadata,
        "tools": None,
    }


def _make_tb_installed_runner(agent_name: str, *, default_env_type: str):
    runner = partial(_run_tb_installed_agent, agent_name, env_type=default_env_type)
    update_wrapper(runner, _run_tb_installed_agent)
    runner.__name__ = f"run_tb_{agent_name.replace('-', '_')}"
    runner.__qualname__ = runner.__name__
    signature = inspect.signature(_run_tb_installed_agent)
    parameters = []
    for parameter in signature.parameters.values():
        if parameter.name == "agent_name":
            continue
        if parameter.name == "env_type":
            parameter = parameter.replace(default=default_env_type)
        parameters.append(parameter)
    runner.__signature__ = signature.replace(parameters=parameters)
    runner.__doc__ = (
        f"Run the {agent_name} installed agent for Terminal Bench "
        f"(default env_type={default_env_type!r})."
    )
    return runner


run_tb_claude_code = _make_tb_installed_runner(
    "claude-code",
    default_env_type="docker",
)
run_tb_qwen_code = _make_tb_installed_runner(
    "qwen-code",
    default_env_type="docker",
)
run_tb_opencode = _make_tb_installed_runner(
    "opencode",
    default_env_type="docker",
)
