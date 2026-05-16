"""OSWorld task adapter."""

import datetime
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from nanorollout.runner import (
    TaskAdapter,
    TaskRunRequest,
    TaskSpec,
)
from nanorollout.harness.agents.shared.llm_config import parse_json_object
from nanorollout.adapters.osworld.task import load_osworld_task, resolve_test_all_path

logger = logging.getLogger(__name__)

OSWorldAgentBuilder = Callable[[TaskSpec, TaskRunRequest], Any]


def _resolved_status(reward: float) -> str:
    if reward >= 1.0:
        return "FULL"
    if reward > 0:
        return "PARTIAL"
    return "NO"


def _terminate_orphan_vm(
    vm_id: Optional[str],
    *,
    region: str,
    instance_id: str,
) -> bool:
    if not vm_id:
        return False
    try:
        import boto3

        boto3.client("ec2", region_name=region).terminate_instances(InstanceIds=[vm_id])
        logger.warning("[%s] Terminated orphan EC2 %s", instance_id, vm_id)
        return True
    except Exception:
        logger.exception("[%s] Failed to terminate orphan EC2 %s", instance_id, vm_id)
        return False


class OSWorldTaskAdapter(TaskAdapter):
    runner_label = "OSWorld"

    def __init__(self, agent_builder: OSWorldAgentBuilder) -> None:
        self.agent_builder = agent_builder

    def prepare_task(self, request: TaskRunRequest, trial_dir: Path) -> TaskSpec:
        extra_args = request.extra_args
        sampling_params = parse_json_object(request.sampling_params)
        test_all_path = resolve_test_all_path(extra_args)
        task = load_osworld_task(request.instance_id, test_all_path)
        instruction = task["instruction"]
        domain = task.get("_domain", "unknown")
        logger.info(
            "[%s] domain=%s instruction=%s",
            request.instance_id,
            domain,
            instruction[:80],
        )
        (trial_dir / "task_info.json").write_text(
            json.dumps(
                {
                    "instance_id": request.instance_id,
                    "domain": domain,
                    "instruction": instruction,
                    "evaluator": task.get("evaluator", {}),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return TaskSpec(
            id=request.instance_id,
            kind="osworld",
            payload=task,
            instruction=instruction,
            environment={
                "observation_type": extra_args.get("observation_type", "screenshot"),
                "sleep_after_execution": extra_args.get("sleep_after_execution", 3),
            },
            evaluation={
                "trajectory": [],
                "reward": 0.0,
            },
            metadata={
                "instance_id": request.instance_id,
                "agent_name": extra_args.get("agent", "qwen3vl"),
                "domain": domain,
                "max_steps": extra_args.get(
                    "max_steps",
                    extra_args.get("max_iterations", 15),
                ),
                "sampling_params": sampling_params,
                "timings": {},
                "trial_dir": trial_dir,
                "steps_taken": 0,
                "done": False,
            },
        )

    def create_environment(
        self,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> Any:
        from nanorollout.envs.desktop_env.desktop_env import DesktopEnv

        timings = task.metadata["timings"]
        observation_type = task.environment["observation_type"]
        t0 = time.time()
        env = DesktopEnv(
            provider_name=request.env_type,
            region=request.extra_args.get("region", "us-east-1"),
            os_type="Ubuntu",
            action_space="pyautogui",
            headless=True,
            require_a11y_tree=observation_type in ("a11y_tree", "screenshot_a11y_tree"),
            require_terminal=False,
            screen_size=(
                int(request.extra_args.get("screen_width", 1920)),
                int(request.extra_args.get("screen_height", 1080)),
            ),
            client_password=request.extra_args.get("client_password", ""),
        )
        timings["ec2_launch_s"] = round(time.time() - t0, 2)
        task.environment["vm_id"] = getattr(env, "path_to_vm", None)
        logger.info(
            "[%s] EC2 worker=%s ip=%s launch=%.1fs",
            request.instance_id,
            getattr(env, "path_to_vm", None),
            getattr(env, "vm_ip", None),
            timings["ec2_launch_s"],
        )
        return env

    def setup_environment(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
    ) -> None:
        timings = task.metadata["timings"]
        t0 = time.time()
        env_obj.reset(task_config=task.payload)
        timings["env_reset_s"] = round(time.time() - t0, 2)
        logger.info(
            "[%s] env.reset done %.1fs",
            request.instance_id,
            timings["env_reset_s"],
        )
        time.sleep(request.extra_args.get("wait_after_reset", 5))

        t0 = time.time()
        task.environment["obs"] = env_obj._get_obs()
        timings["first_obs_s"] = round(time.time() - t0, 2)

        try:
            env_obj.controller.start_recording()
        except Exception:
            pass

    def build_agent(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> Any:
        del env_obj, trial_dir
        agent = self.agent_builder(task, request)
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()
        return agent

    def run_agent(
        self,
        agent: Any,
        task: TaskSpec,
        env_obj: Any,
    ) -> Any:
        obs = task.environment["obs"]
        done = False
        steps_taken = 0
        total_predict_time = 0.0
        total_env_step_time = 0.0
        total_screenshot_time = 0.0
        trajectory = task.evaluation["trajectory"]
        timings = task.metadata["timings"]
        trial_dir = task.metadata["trial_dir"]

        while not done and steps_taken < task.metadata["max_steps"]:
            t0 = time.time()
            response, actions = agent.predict(task.instruction, obs)
            predict_time = time.time() - t0
            total_predict_time += predict_time
            logger.info(
                "[%s] step=%d predict=%.1fs actions=%s",
                task.metadata["instance_id"],
                steps_taken + 1,
                predict_time,
                actions,
            )

            if not actions:
                obs = env_obj._get_obs()
                steps_taken += 1
                continue

            for action in actions:
                ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                t0 = time.time()
                obs, step_reward, done, info = env_obj.step(
                    action,
                    task.environment["sleep_after_execution"],
                )
                env_step_time = time.time() - t0
                total_env_step_time += env_step_time
                trajectory.append(
                    {
                        "step_num": steps_taken + 1,
                        "action_timestamp": ts,
                        "action": action,
                        "response": response
                        if isinstance(response, str)
                        else str(response),
                        "predict_time_s": round(predict_time, 2),
                        "env_step_time_s": round(env_step_time, 2),
                        "reward": step_reward,
                        "done": done,
                        "info": info,
                    }
                )

                t0 = time.time()
                try:
                    (trial_dir / f"step_{steps_taken + 1}_{ts}.png").write_bytes(
                        obs["screenshot"]
                    )
                except Exception:
                    pass
                total_screenshot_time += time.time() - t0
                if done:
                    break

            steps_taken += 1

        timings["total_predict_s"] = round(total_predict_time, 2)
        timings["total_env_step_s"] = round(total_env_step_time, 2)
        timings["total_screenshot_save_s"] = round(total_screenshot_time, 2)
        task.metadata["steps_taken"] = steps_taken
        task.metadata["done"] = done
        messages = [
            {"role": "agent", "content": entry.get("response", "")[:500]}
            for entry in trajectory
        ]
        return SimpleNamespace(
            history=messages,
            success=True,
            message="",
            iterations=steps_taken,
            error=None,
            exit_status="finished" if done else "max_steps",
            raw={**task.metadata, **task.environment, **task.evaluation},
        )

    def evaluate(
        self,
        env_obj: Any,
        task: TaskSpec,
        request: TaskRunRequest,
        trial_dir: Path,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        del trial_dir
        time.sleep(request.extra_args.get("wait_before_eval", 5))
        t0 = time.time()
        reward = float(env_obj.evaluate())
        task.metadata["timings"]["eval_s"] = round(time.time() - t0, 2)
        task.evaluation["reward"] = reward
        payload = {
            "resolved": reward > 0,
            "resolved_status": _resolved_status(reward),
            "reward": reward,
        }
        logger.info(
            "[%s] reward=%.2f exit_status=%s steps=%d",
            request.instance_id,
            reward,
            "finished" if task.metadata["done"] else "max_steps",
            task.metadata["steps_taken"],
        )
        return payload, None

    def build_reward_payload(
        self,
        instance_id: str,
        eval_payload: Dict[str, Any],
        error_msg: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "instance_id": instance_id,
            "resolved": eval_payload.get("resolved", False),
            "resolved_status": eval_payload.get("resolved_status", "NO"),
            "reward": eval_payload.get("reward", 0.0),
            "error": error_msg,
        }

    def stop_environment(
        self,
        env_obj: Any,
        task: Optional[TaskSpec],
        request: TaskRunRequest,
    ) -> None:
        if task is not None:
            try:
                env_obj.controller.end_recording(
                    str(task.metadata["trial_dir"] / "recording.mp4")
                )
            except Exception:
                pass
        try:
            env_obj.close()
            logger.info("[%s] EC2 terminated via env.close()", request.instance_id)
        except Exception:
            logger.exception("[%s] env.close() failed", request.instance_id)
            vm_id = None
            if task is not None:
                vm_id = task.environment.get("vm_id")
            vm_id = vm_id or getattr(env_obj, "path_to_vm", None)
            terminated = _terminate_orphan_vm(
                vm_id,
                region=request.extra_args.get("region", "us-east-1"),
                instance_id=request.instance_id,
            )
            if not terminated:
                raise

    def build_agent_metrics(
        self,
        messages: list[Dict[str, Any]],
        agent_time: float,
        eval_time: float,
        total_time: float,
        agent_result: Any,
        eval_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        del messages, agent_time, eval_payload
        raw = getattr(agent_result, "raw", {}) if agent_result else {}
        timings = raw.get("timings", {})
        return {
            "turns": raw.get("steps_taken", 0),
            "tool_calls": len(raw.get("trajectory", [])),
            "model_query_time_sum": timings.get("total_predict_s", 0),
            "env_execution_time_sum": timings.get("total_env_step_s", 0),
            "eval_time": timings.get("eval_s", eval_time),
            "agent_run_time": total_time,
            "total_time": total_time,
        }

    def update_metadata(
        self,
        metadata: Dict[str, Any],
        task: Optional[TaskSpec],
        agent_result: Any,
        eval_payload: Dict[str, Any],
        error_msg: Optional[str],
    ) -> Dict[str, Any]:
        del agent_result, error_msg
        if task is None:
            return metadata
        metadata.update(
            {
                "agent": task.metadata["agent_name"],
                "resolved": eval_payload.get("resolved"),
                "resolved_status": eval_payload.get("resolved_status"),
                "timings": task.metadata["timings"],
            }
        )
        return metadata

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
        del instance_id, model, base_url, env_type, tools_json, eval_output
        raw = getattr(agent_result, "raw", {}) if agent_result else {}
        timings = dict(raw.get("timings", {}))
        timings["total_s"] = round(time.time() - started, 2)
        with open(trial_dir / "traj.jsonl", "w", encoding="utf-8") as handle:
            for entry in raw.get("trajectory", []):
                handle.write(json.dumps(entry, default=str) + "\n")
        (trial_dir / "result.txt").write_text(
            f"{reward_payload['reward']}\n",
            encoding="utf-8",
        )
        (trial_dir / "timings.json").write_text(
            json.dumps(timings, indent=2),
            encoding="utf-8",
        )
        (trial_dir / "reward.json").write_text(
            json.dumps(reward_payload, indent=2),
            encoding="utf-8",
        )
        (trial_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

    def build_exit_status(
        self,
        error_msg: Optional[str],
        agent_result: Any,
        eval_payload: Dict[str, Any],
    ) -> str:
        del eval_payload
        if error_msg:
            return "error"
        return getattr(agent_result, "exit_status", "error")
