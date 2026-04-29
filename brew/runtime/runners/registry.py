from tinyflow.runtime.runners.base import BaseRunner, RunnerSpec
from tinyflow.runtime.runners.oh_core import OHCoreRunner
from tinyflow.runtime.runners.oh_lite import OHLiteRunner
from tinyflow.runtime.runners.miniswe import MiniSweAgentRunner
from tinyflow.runtime.runners.r2egym import R2EGymRunner


class RunnerRegistry:
    @staticmethod
    def create(spec: RunnerSpec) -> BaseRunner:
        runner_type = spec.runner_type.lower()
        if runner_type == "minisweagent":
            return MiniSweAgentRunner(spec)
        if runner_type in {"oh-core", "openhands"}:
            return OHCoreRunner(spec)
        if runner_type == "oh-lite":
            return OHLiteRunner(spec)
        if runner_type in {"r2egym", "r2e-gym"}:
            return R2EGymRunner(spec)
        raise ValueError(f"Unsupported runner type: {runner_type}")
