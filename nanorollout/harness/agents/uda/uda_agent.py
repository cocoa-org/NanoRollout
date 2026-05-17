"""UDA Agent implementation.

Thin wrapper around :class:`nanorollout.envs.uda_env.TaskExecutor`,
which orchestrates the rollout loop on uda-desktop's unified
``/v1/*`` surface (computer-use + shell + file + code + jupyter).

Mirrors the shape of :class:`nanorollout.harness.agents.cocoa.CocoaAgent`
so that the existing runner / scheduler / Ray-server machinery treats
``UDAAgent`` identically.
"""

from typing import Any, Dict

from nanorollout.envs.uda_env import TaskExecutor

from .base import BaseAgent


class UDAAgent(BaseAgent):
    """UDA Agent using the uda_env TaskExecutor."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.executor = TaskExecutor(config)

    def setup_environment(self, task: Dict[str, Any], wait_time: int = 30) -> None:
        """Bring up the uda-desktop sandbox for ``task``."""
        self.executor.setup_environment(task, wait_time=wait_time)

    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task with the configured controller + sandbox."""
        result = self.executor.run_task(task)
        result["agent_type"] = "uda"

        if "task_result" in result:
            result["answer"] = result["task_result"]
        else:
            result["answer"] = ""

        result["trajectory"] = {
            "conversation": result.get("conversation", []),
            "execution_trace": result.get("execution_trace", []),
            "visualization_data": result.get("visualization_data", {}),
        }

        return result

    def run_eval(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task's bundled verifier (``test.py``-style) on ``result``."""
        return self.executor.run_eval(task, result)

    def cleanup_environment(self) -> None:
        """Tear down the uda-desktop sandbox."""
        self.executor.cleanup_environment()
