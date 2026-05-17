"""osworld-v1 driver.

OSWorld v1 is fundamentally different from cocoa-v1 / wildclaw-v1: there
is no uda-desktop container to stage files into, no exec/ + gt/ pair to
copy. The OSWorld VM brings up its own setup primitives (chrome
open-tabs, launch, googledrive, ...) inside ``env.reset(task_config)``
and bundles its evaluator into ``env.evaluate()``.

This driver therefore reduces to two real responsibilities:

* ``load_task`` — read the task JSON from the bundled OSWorld corpus at
  ``examples/eval/osworld/data/`` and attach it as ``_osworld_config``
  for :class:`OSWorldV1Adapter` to consume in ``create_environment``.
* ``score`` — call ``runtime.evaluate()`` (the live DesktopEnv handle
  the adapter exposes as ``self.runtime``) and wrap the result as
  ``{overall_score: float, success: bool}`` to match wildclaw / cocoa
  score-dict shape.

All other Protocol methods are no-ops: OSWorld owns its own workspace
setup, warmup, and ground-truth injection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("uda.driver.osworld_v1")


class OSWorldV1Driver:
    """Driver for ``adapter/osworld-v1/`` tasks (corpus lives in examples/)."""

    name = "osworld-v1"
    # OSWorld doesn't have a single canonical "task workspace" path — apps
    # write to /home/user/Desktop, /tmp, /root, ... per task. Kept here for
    # interface compatibility; not used by this driver.
    container_workspace = "/root"

    # ----- offline / host-side ------------------------------------------

    def load_task(self, task_dir: Path) -> Dict[str, Any]:
        """Read the OSWorld task JSON.

        ``task_dir`` is one of:

        1. A direct path to ``examples/eval/osworld/data/examples/<domain>/<id>.json``
           (passed as a file, not a directory — supported for ad-hoc use).
        2. A directory containing a ``meta.json`` with ``"osworld_config_path"``
           pointing at the canonical OSWorld JSON.
        """
        if task_dir.is_file() and task_dir.suffix == ".json":
            osworld_config = json.loads(task_dir.read_text(encoding="utf-8"))
        else:
            meta_path = task_dir / "meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(
                    f"osworld-v1: {task_dir} has no meta.json and is not a .json file"
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            config_path = meta.get("osworld_config_path")
            if not config_path:
                raise ValueError(
                    f"osworld-v1: {meta_path} is missing 'osworld_config_path'"
                )
            osworld_config = json.loads(Path(config_path).read_text(encoding="utf-8"))

        if not isinstance(osworld_config, dict) or "instruction" not in osworld_config:
            raise ValueError(
                f"osworld-v1: malformed task config (no 'instruction'): {task_dir}"
            )

        task: Dict[str, Any] = {
            "driver": self.name,
            "task_dir": str(task_dir),
            "task_name": osworld_config.get("id", task_dir.name),
            "task_id": osworld_config.get("id", task_dir.name),
            "instruction": osworld_config["instruction"],
            "timeout_seconds": int(osworld_config.get("timeout_seconds", 1800)),
            "extra_env": {},
            "_osworld_config": osworld_config,
        }
        return task

    def get_extra_env(self, task: Dict[str, Any]) -> Dict[str, str]:
        """OSWorld doesn't ship per-task env.tsv; nothing to inject."""
        return {}

    # ----- in-container, pre-rollout ------------------------------------

    def setup_workspace(self, runtime: Any, task: Dict[str, Any]) -> None:
        """No-op: OSWorld's setup primitives run inside ``env.reset(task_config)``."""
        return None

    def run_warmup(self, runtime: Any, task: Dict[str, Any]) -> None:
        """No-op: ditto setup_workspace."""
        return None

    # ----- in-container, post-rollout -----------------------------------

    def inject_ground_truth(self, runtime: Any, task: Dict[str, Any]) -> None:
        """No-op: OSWorld evaluators read state directly from the VM."""
        return None

    def score(
        self,
        runtime: Any,
        task: Dict[str, Any],
        rollout_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Invoke OSWorld's bundled evaluator.

        ``runtime`` is the live ``DesktopEnv`` that
        :class:`OSWorldV1Adapter` stashes there. We support either a
        bare DesktopEnv or anything else with an ``evaluate()`` method
        (so the same driver path works in tests with stub runtimes).
        """
        if runtime is None or not hasattr(runtime, "evaluate"):
            logger.error(
                "osworld-v1: runtime %r has no .evaluate() — was the env launched?",
                runtime,
            )
            return None
        score = float(runtime.evaluate())
        return {
            "overall_score": score,
            "success": score >= 1.0,
        }
