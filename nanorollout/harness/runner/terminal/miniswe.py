"""Terminal Bench MiniSweAgent runner entry point."""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from nanorollout.harness.runner.terminal.common import (
    DEFAULT_TERMINAL_BENCH_REPO_URL,
    _build_agent_config,
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
    resolve_tb_repo_dir,
    resolve_tb_instance,
    resolve_tb_timeouts,
    trial_logging,
)

logger = logging.getLogger(__name__)


def run_tb_miniswe(
    instance_id: str,
    output_dir: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    env_type: str = "docker",
    sampling_params: Optional[object] = None,
    extra_args: Dict[str, Any] = {},
) -> Dict[str, Any]:

    from nanorollout.eval.terminalbench.grading import run_tb_eval

    for key in ("env_timeout", "create_timeout", "max_iterations"):
        if key not in extra_args:
            raise ValueError(f"Missing required env argument: {key}")

    # extra env args
    step_timeout = extra_args.get("step_timeout")
    eval_timeout = extra_args.get("eval_timeout")
    max_iterations = extra_args["max_iterations"]
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
    tools_json = None

    output_root = Path(output_dir)
    trial_dir = output_root
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        instance = resolve_tb_instance(
            instance_id=instance_id,
            repo_dir=repo_dir,
        )
        image_name = ensure_tb_image(instance, env_type=env_type)

        # Resolve per-task timeouts from task.toml (aligned with Harbor)
        agent_timeout, resolved_eval_timeout = resolve_tb_timeouts(instance, extra_args)
        logger.info(
            "[%s] Per-task timeouts: agent=%.0fs, eval=%.0fs",
            instance_id,
            agent_timeout,
            resolved_eval_timeout,
        )
        # Use resolved timeouts, fall back to extra_args value for env creation
        env_step_timeout = step_timeout or agent_timeout
        env_eval_timeout = eval_timeout or resolved_eval_timeout

        api_key = api_key or "abc-123"
        agent_config = _build_agent_config(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_iterations=max_iterations,
            sampling_params=sampling_params,
            default_temperature=0.6,
            default_top_p=0.95,
        )

        with env_logging(trial_dir):
            with trial_logging(trial_dir):
                logger.info(
                    "[%s] Starting %s container %s",
                    instance_id,
                    env_type,
                    image_name,
                )
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
                env_obj.set_tool_log_context(f"{instance_id}")
                env_obj.start()

                instruction = instance["instruction"]

                logger.info("[%s] Running TerminalMiniSweAgent", instance_id)
                agent_start = time.time()
                from nanorollout.harness.agents.terminal.miniswe import TerminalMiniSweAgent

                agent = TerminalMiniSweAgent(
                    environment=env_obj,
                    config=agent_config,
                    step_timeout=env_step_timeout,
                )
                tools_schema = agent.get_tools_schema()
                tools_json = tools_schema if tools_schema else None
                agent_result = agent.run(instruction)
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
        logger.exception("TerminalMiniSweAgent run failed for %s", instance_id)
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

    if trial_dir:
        _write_artifacts(
            trial_dir=trial_dir,
            instance_id=instance_id,
            model=model_name,
            base_url=base_url,
            env_type=env_type,
            agent_result=agent_result,
            tools_json=tools_json,
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

    exit_status = "Error" if error_msg else _resolve_exit_status(eval_payload)
    gc.collect()
    return {
        "reward": reward_payload.get("reward", 0),
        "messages": messages,
        "exit_status": exit_status,
        "agent_metrics": agent_metrics,
        "metadata": metadata,
        "tools": tools_json,
    }
