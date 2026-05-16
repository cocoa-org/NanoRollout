"""cocoa-v1 driver.

Encapsulates the legacy CocoaBench-v1.0 contract that ``TaskExecutor``
originally hard-coded:

- ``task.yaml.enc`` + ``test.py.enc`` + ``canary.txt`` in every task dir
- Inputs (PDFs / images / repos) are baked into the per-task Docker image
- Verifier runs **host-side**: decrypt ``test.py.enc`` with the canary,
  exec it, call ``test(rollout_result)``

Setup / warmup / GT-injection are all no-ops because everything is
pre-staged at image-build time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml

from ..decrypt import decrypt_file_to_memory, read_canary
from ..logger import get_logger
from .base import discover_workspace_assets

if TYPE_CHECKING:
    from ..base import BaseSandboxRuntime

logger = get_logger("uda.driver.cocoa_v1")


class CocoaV1Driver:
    """Driver for ``adapter/cocoa-v1/`` tasks.

    The legacy cocoa pipeline expected:

    ::

        task_dir/
          task.yaml.enc
          test.py.enc
          canary.txt
          Dockerfile          # bakes all input data into the image
          docker-compose.yaml

    This driver simply replays that pipeline through the new protocol.
    All in-container methods are no-ops because cocoa's task images
    already contain their workspace at build time.
    """

    name = "cocoa-v1"
    container_workspace = "/home/kasm-user"

    # --- Offline / host-side ---------------------------------------------

    def load_task(self, task_dir: Path) -> Dict[str, Any]:
        assets = discover_workspace_assets(task_dir)

        encrypted = "task_yaml_enc" in assets and "task_yaml" not in assets
        if encrypted:
            canary = read_canary(task_dir)
            if canary is None:
                raise ValueError(
                    f"cocoa-v1: {task_dir} has task.yaml.enc but no canary.txt"
                )
            task_yaml = decrypt_file_to_memory(task_dir / "task.yaml.enc", canary)
            test_file = task_dir / "test.py.enc"
        else:
            with (task_dir / "task.yaml").open(encoding="utf-8") as fh:
                task_yaml = fh.read()
            test_file = task_dir / "test.py"

        task_data = yaml.safe_load(task_yaml)
        if not isinstance(task_data, dict):
            raise ValueError(f"cocoa-v1: task.yaml in {task_dir} must be a mapping")

        task = dict(task_data)
        task["driver"] = self.name
        task["task_dir"] = str(task_dir)
        task["task_name"] = task_dir.name
        task["use_encrypted"] = encrypted
        task["test_file_path"] = str(test_file) if test_file.exists() else None
        # ``instruction`` is the canonical agent-visible prompt; cocoa
        # task.yaml typically already uses that key.
        task.setdefault("instruction", task_data.get("instruction") or task_data.get("prompt", ""))
        return task

    def get_extra_env(self, task: Dict[str, Any]) -> Dict[str, str]:
        # cocoa task.yaml may declare an `env:` block; pass through whatever
        # plain string env spec it contains. Empty by default.
        env = task.get("env", "")
        if isinstance(env, dict):
            return {k: str(v) for k, v in env.items()}
        return {}

    # --- In-container, pre-rollout (no-ops for cocoa) --------------------

    def setup_workspace(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any]
    ) -> None:
        # Inputs are baked into the per-task Docker image at build time
        # — no host-to-container copy needed. See cocoa task Dockerfiles.
        logger.debug(
            "cocoa-v1: setup_workspace is a no-op for %s (assets baked into image)",
            task.get("task_name"),
        )

    def run_warmup(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any]
    ) -> None:
        logger.debug(
            "cocoa-v1: run_warmup is a no-op for %s (Dockerfile handles setup)",
            task.get("task_name"),
        )

    # --- In-container, post-rollout --------------------------------------

    def inject_ground_truth(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any]
    ) -> None:
        # cocoa's GT lives inside the canary-XOR'd test.py.enc closure;
        # there is nothing to copy into the container.
        logger.debug(
            "cocoa-v1: inject_ground_truth is a no-op for %s (GT inside test.py.enc)",
            task.get("task_name"),
        )

    def score(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any],
        rollout_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        test_file_path = task.get("test_file_path")
        if not test_file_path:
            logger.debug(
                "cocoa-v1: no test file for %s, skipping eval",
                task.get("task_name"),
            )
            return None

        test_file = Path(test_file_path)
        if not test_file.exists():
            logger.warning("cocoa-v1: test file %s missing", test_file_path)
            return None

        if task.get("use_encrypted"):
            canary = read_canary(Path(task["task_dir"]))
            if canary is None:
                raise ValueError(
                    f"cocoa-v1: encrypted test in {test_file} but no canary.txt"
                )
            code = decrypt_file_to_memory(test_file, canary)
            module = type(sys)("test")
            module.__file__ = str(test_file)
            sys.modules["test"] = module
            exec(compile(code, str(test_file), "exec"), module.__dict__)
        else:
            spec = importlib.util.spec_from_file_location("test", test_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["test"] = module
            spec.loader.exec_module(module)

        test_fn = getattr(module, "test", None)
        if test_fn is None:
            logger.warning("cocoa-v1: %s has no test() function", test_file)
            return None
        return test_fn(rollout_result)
