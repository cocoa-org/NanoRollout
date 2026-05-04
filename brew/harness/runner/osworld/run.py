"""OSWorld runner entry point.

Minimal orchestrator: env lifecycle + agent dispatch + result packaging.
Agent logic lives in brew/harness/agents/osworld/<agent_name>.py.
"""
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)

OSWORLD_DATA_ROOT = os.environ.get(
    "OSWORLD_ROOT",
    "/mnt/weka/home/zhuojun.cheng/uda-org/OSWorld",
)

# Agent registry — add new agents here
AGENT_REGISTRY = {
    "qwen3vl": "brew.harness.agents.osworld.qwen3vl.Qwen3VLAgent",
}


def _load_task(task_id: str, test_all_meta_path: str) -> Dict[str, Any]:
    with open(test_all_meta_path, "r") as f:
        test_all = json.load(f)
    for domain, ids in test_all.items():
        if task_id in ids:
            config_path = os.path.join(
                os.path.dirname(test_all_meta_path), "examples", domain, f"{task_id}.json",
            )
            with open(config_path, "r") as f:
                task = json.load(f)
            task["_domain"] = domain
            return task
    raise ValueError(f"Task {task_id} not found in {test_all_meta_path}")


def _create_agent(agent_name: str, **kwargs):
    """Instantiate an agent by name from the registry."""
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(AGENT_REGISTRY.keys())}")
    module_path, class_name = AGENT_REGISTRY[agent_name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def run_osworld(
    instance_id: str,
    output_dir: str,
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    env_type: str = "aws",
    sampling_params: Optional[object] = None,
    extra_args: Dict[str, Any] = {},
) -> Dict[str, Any]:
    from brew.envs.desktop_env.desktop_env import DesktopEnv

    if isinstance(sampling_params, str):
        try:
            sampling_params = json.loads(sampling_params)
        except json.JSONDecodeError:
            sampling_params = {}
    if not isinstance(sampling_params, dict):
        sampling_params = {}

    max_steps = extra_args.get("max_steps", extra_args.get("max_iterations", 15))
    sleep_after_execution = extra_args.get("sleep_after_execution", 3)
    observation_type = extra_args.get("observation_type", "screenshot")
    agent_name = extra_args.get("agent", "qwen3vl")

    test_all_path = extra_args.get(
        "test_all_meta_path",
        os.path.join(OSWORLD_DATA_ROOT, "evaluation_examples", "test_all.json"),
    )

    started = time.time()
    env = None
    _instance_vm_id = None
    error_msg = None
    reward = 0.0
    trajectory: List[Dict] = []
    timings: Dict[str, float] = {}
    exit_status = "error"
    steps_taken = 0

    trial_dir = Path(output_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        task = _load_task(instance_id, test_all_path)
        instruction = task["instruction"]
        domain = task.get("_domain", "unknown")
        logger.info("[%s] domain=%s instruction=%s", instance_id, domain, instruction[:80])

        # Save task info
        with open(trial_dir / "task_info.json", "w") as f:
            json.dump({
                "instance_id": instance_id,
                "domain": domain,
                "instruction": instruction,
                "evaluator": task.get("evaluator", {}),
            }, f, indent=2, ensure_ascii=False)

        # Create agent
        agent = _create_agent(
            agent_name,
            model=model_name,
            max_tokens=sampling_params.get("max_tokens", 32768),
            top_p=sampling_params.get("top_p", 0.9),
            temperature=sampling_params.get("temperature", 0.0),
            action_space="pyautogui",
            observation_type=observation_type,
            history_n=extra_args.get("history_n", 4),
            coordinate_type=extra_args.get("coordinate_type", "relative"),
            base_url=base_url,
            api_key=api_key,
        )
        agent.reset()

        # --- EC2 launch ---
        t0 = time.time()
        env = DesktopEnv(
            provider_name=env_type,
            region=extra_args.get("region", "us-east-1"),
            os_type="Ubuntu",
            action_space="pyautogui",
            headless=True,
            require_a11y_tree=(observation_type in ("a11y_tree", "screenshot_a11y_tree")),
            require_terminal=False,
            screen_size=(
                int(extra_args.get("screen_width", 1920)),
                int(extra_args.get("screen_height", 1080)),
            ),
        )
        timings["ec2_launch_s"] = round(time.time() - t0, 2)
        _instance_vm_id = getattr(env, "path_to_vm", None)
        logger.info("[%s] EC2 worker=%s ip=%s (launch %.1fs)", instance_id, env.path_to_vm, env.vm_ip, timings["ec2_launch_s"])

        # --- env.reset (task setup) ---
        t0 = time.time()
        env.reset(task_config=task)
        timings["env_reset_s"] = round(time.time() - t0, 2)
        logger.info("[%s] env.reset done (%.1fs)", instance_id, timings["env_reset_s"])

        time.sleep(extra_args.get("wait_after_reset", 5))

        t0 = time.time()
        obs = env._get_obs()
        timings["first_obs_s"] = round(time.time() - t0, 2)

        done = False
        try:
            env.controller.start_recording()
        except Exception:
            pass

        # --- Agent loop ---
        total_predict_time = 0.0
        total_env_step_time = 0.0
        total_screenshot_time = 0.0

        while not done and steps_taken < max_steps:
            # predict
            t0 = time.time()
            response, actions = agent.predict(instruction, obs)
            predict_time = time.time() - t0
            total_predict_time += predict_time

            logger.info("[%s] step=%d predict=%.1fs actions=%s", instance_id, steps_taken + 1, predict_time, actions)

            if not actions:
                obs = env._get_obs()
                steps_taken += 1
                continue

            for action in actions:
                ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")

                # env.step
                t0 = time.time()
                obs, step_reward, done, info = env.step(action, sleep_after_execution)
                env_step_time = time.time() - t0
                total_env_step_time += env_step_time

                trajectory.append({
                    "step_num": steps_taken + 1,
                    "action_timestamp": ts,
                    "action": action,
                    "response": response if isinstance(response, str) else str(response),
                    "predict_time_s": round(predict_time, 2),
                    "env_step_time_s": round(env_step_time, 2),
                    "reward": step_reward,
                    "done": done,
                    "info": info,
                })

                # save screenshot
                t0 = time.time()
                try:
                    (trial_dir / f"step_{steps_taken + 1}_{ts}.png").write_bytes(obs["screenshot"])
                except Exception:
                    pass
                total_screenshot_time += time.time() - t0

                if done:
                    break

            steps_taken += 1

        timings["total_predict_s"] = round(total_predict_time, 2)
        timings["total_env_step_s"] = round(total_env_step_time, 2)
        timings["total_screenshot_save_s"] = round(total_screenshot_time, 2)

        # --- evaluate ---
        time.sleep(extra_args.get("wait_before_eval", 5))
        t0 = time.time()
        reward = float(env.evaluate())
        timings["eval_s"] = round(time.time() - t0, 2)

        exit_status = "finished" if done else "max_steps"
        logger.info("[%s] reward=%.2f exit_status=%s steps=%d", instance_id, reward, exit_status, steps_taken)
        logger.info("[%s] timings: %s", instance_id, json.dumps(timings))

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("[%s] Error: %s", instance_id, exc)
        exit_status = "error"

    finally:
        if env is not None:
            try:
                env.controller.end_recording(str(trial_dir / "recording.mp4"))
            except Exception:
                pass
            try:
                env.close()
                logger.info("[%s] EC2 terminated via env.close()", instance_id)
            except Exception as e:
                logger.warning("[%s] env.close() failed: %s", instance_id, e)
        elif _instance_vm_id:
            try:
                import boto3
                region = extra_args.get("region", "us-east-1")
                boto3.client("ec2", region_name=region).terminate_instances(InstanceIds=[_instance_vm_id])
                logger.warning("[%s] Terminated orphan EC2 %s", instance_id, _instance_vm_id)
            except Exception as e:
                logger.error("[%s] Failed to terminate orphan EC2 %s: %s", instance_id, _instance_vm_id, e)

    total_time = time.time() - started
    timings["total_s"] = round(total_time, 2)

    try:
        with open(trial_dir / "traj.jsonl", "w") as f:
            for entry in trajectory:
                f.write(json.dumps(entry, default=str) + "\n")
        (trial_dir / "result.txt").write_text(f"{reward}\n")
        with open(trial_dir / "timings.json", "w") as f:
            json.dump(timings, f, indent=2)
    except Exception as e:
        logger.warning("[%s] Failed to write artifacts: %s", instance_id, e)

    return {
        "reward": reward,
        "messages": [{"role": "agent", "content": e.get("response", "")[:500]} for e in trajectory],
        "exit_status": exit_status,
        "agent_metrics": {
            "turns": steps_taken,
            "tool_calls": len(trajectory),
            "model_query_time_sum": timings.get("total_predict_s", 0),
            "env_execution_time_sum": timings.get("total_env_step_s", 0),
            "eval_time": timings.get("eval_s", 0),
            "agent_run_time": total_time,
            "total_time": total_time,
        },
        "metadata": {
            "instance_id": instance_id,
            "environment": env_type,
            "agent": agent_name,
            "resolved": reward > 0,
            "resolved_status": "FULL" if reward >= 1.0 else ("PARTIAL" if reward > 0 else "NO"),
            "error": error_msg,
            "trial_dir": str(trial_dir),
            "exit_status": exit_status,
            "timings": timings,
        },
    }
