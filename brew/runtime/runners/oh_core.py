from typing import Any, Dict

from tinyflow.runtime.runners.base import BaseRunner, RunnerSpec


class OHCoreRunner(BaseRunner):
    DEFAULT_MODULE = "harness/tinyflow_runner/oh_core.py"
    DEFAULT_ENTRYPOINT = "run_oh_core"

    def __init__(self, spec: RunnerSpec) -> None:
        super().__init__(spec)
        module_name = spec.runner_module or self.DEFAULT_MODULE
        entrypoint = spec.runner_entrypoint or self.DEFAULT_ENTRYPOINT
        self._runner = self._load_runner(module_name, entrypoint)

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        assert "output_dir" in params
        assert "extra_args" in params
        return self._runner(**params)
