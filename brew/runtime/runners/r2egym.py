from typing import Any, Dict

from tinyflow.runtime.runners.base import BaseRunner, RunnerSpec


class R2EGymRunner(BaseRunner):
    DEFAULT_MODULE = "harness/tinyflow_runner/r2egym.py"
    DEFAULT_ENTRYPOINT = "run_r2egym"

    def __init__(self, spec: RunnerSpec) -> None:
        super().__init__(spec)
        module_name = spec.runner_module or self.DEFAULT_MODULE
        entrypoint = spec.runner_entrypoint or self.DEFAULT_ENTRYPOINT
        self._runner = self._load_runner(module_name, entrypoint)

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        assert "output_dir" in params
        assert "extra_args" in params
        return self._runner(**params)
