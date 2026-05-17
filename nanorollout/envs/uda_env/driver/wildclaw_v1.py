"""wildclaw-v1 driver.

Encapsulates the WildClawBench contract:

- ``task.yaml`` (plain) + ``grade.py`` + ``env.tsv`` + ``warmup.sh``
- ``exec/`` — pre-rollout inputs copied into ``/tmp_workspace/``
- ``gt/``   — post-rollout ground truth copied into ``/tmp_workspace/gt/``
- ``grade.py`` runs **inside the container** with cwd ``/tmp_workspace``,
  reading ``results/`` (agent output) and ``gt/`` (injected late)

The result is a JSON-serialisable dict of float scores, parsed from the
container's stdout.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml

from ..logger import get_logger
from .base import discover_workspace_assets

if TYPE_CHECKING:
    from ..base import BaseSandboxRuntime

logger = get_logger("uda.driver.wildclaw_v1")


class WildclawV1Driver:
    """Driver for ``adapter/wildclaw-v1/`` tasks."""

    name = "wildclaw-v1"
    container_workspace = "/tmp_workspace"

    # --- Offline / host-side ---------------------------------------------

    def load_task(self, task_dir: Path) -> Dict[str, Any]:
        assets = discover_workspace_assets(task_dir)

        if "task_yaml" not in assets:
            raise FileNotFoundError(
                f"wildclaw-v1: {task_dir}/task.yaml missing"
            )
        with assets["task_yaml"].open(encoding="utf-8") as fh:
            task_data = yaml.safe_load(fh)
        if not isinstance(task_data, dict):
            raise ValueError(f"wildclaw-v1: task.yaml in {task_dir} must be a mapping")

        # meta.json gives id / category / timeout_seconds; fall back to dir
        # name + defaults if it's absent.
        meta: Dict[str, Any] = {}
        if "meta" in assets:
            with assets["meta"].open(encoding="utf-8") as fh:
                meta = json.load(fh)

        task = dict(task_data)
        task["driver"] = self.name
        task["task_dir"] = str(task_dir)
        task["task_name"] = task_dir.name
        task["instruction"] = task_data.get("instruction", "")
        task["timeout_seconds"] = meta.get("timeout_seconds") or task_data.get(
            "timeout_seconds", 600
        )
        task["task_id"] = meta.get("id", task_dir.name)
        task["category"] = meta.get("category", "")
        task["grade_file_path"] = (
            str(assets["grade_py"]) if "grade_py" in assets else None
        )
        task["exec_dir"] = str(assets["exec"]) if "exec" in assets else None
        task["gt_dir"] = str(assets["gt"]) if "gt" in assets else None
        task["warmup_path"] = (
            str(assets["warmup_sh"]) if "warmup_sh" in assets else None
        )
        task["env_tsv_path"] = (
            str(assets["env_tsv"]) if "env_tsv" in assets else None
        )
        return task

    def get_extra_env(self, task: Dict[str, Any]) -> Dict[str, str]:
        env_tsv_path = task.get("env_tsv_path")
        if not env_tsv_path:
            return {}
        out: Dict[str, str] = {}
        import os
        for line in Path(env_tsv_path).read_text(encoding="utf-8").splitlines():
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            value = os.environ.get(key, "")
            if value:
                out[key] = value
        return out

    # --- In-container, pre-rollout ---------------------------------------

    def setup_workspace(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any]
    ) -> None:
        ws = self.container_workspace

        # Stage env vars for the AGENT's rollout shell sessions: write a
        # profile.d snippet (so login shells get them) and a /tmp_workspace/.env
        # file (for explicit `source`). grade.py gets env vars via inline
        # prefix in ``score`` regardless of these files — they only matter
        # for the agent's own shell/code execution during rollout.
        extra_env = self.get_extra_env(task)
        if extra_env:
            import shlex as _shlex
            import tempfile
            exports = "\n".join(
                f"export {k}={_shlex.quote(v)}" for k, v in extra_env.items()
            ) + "\n"
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tmp:
                tmp.write(exports)
                tmp_path = tmp.name
            try:
                runtime.copy_to_runtime(tmp_path, "/etc/profile.d/uda_env.sh")
                runtime.copy_to_runtime(tmp_path, f"{ws.rstrip('/')}/.env")
                logger.info(
                    "wildclaw-v1: staged %d env var(s) (profile.d + .env)",
                    len(extra_env),
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        exec_dir = task.get("exec_dir")
        if not exec_dir:
            logger.debug(
                "wildclaw-v1: %s has no exec/ — agent fetches inputs at runtime",
                task.get("task_name"),
            )
            return

        exec_path = Path(exec_dir)
        # Copy each top-level file/dir under exec/ into self.container_workspace so the
        # agent sees them at /tmp_workspace/<name>. Idempotent: re-runs
        # overwrite. Uses the runtime's existing ``copy_to_runtime`` so
        # docker / modal both work.
        for entry in sorted(exec_path.iterdir()):
            dest = f"{ws.rstrip('/')}/{entry.name}"
            ok = runtime.copy_to_runtime(str(entry), dest)
            if not ok:
                raise RuntimeError(
                    f"wildclaw-v1: setup_workspace failed for {entry} → {dest}"
                )
            logger.info(
                "wildclaw-v1: staged %s → %s (container)",
                entry.name,
                dest,
            )

    def run_warmup(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any]
    ) -> None:
        warmup_path = task.get("warmup_path")
        if not warmup_path:
            return
        warmup_text = Path(warmup_path).read_text(encoding="utf-8")
        # Heuristic: warmup is meaningful only if it has non-comment / non-
        # empty lines beyond the standard ``#!/usr/bin/env bash`` + ``set -e``
        # header. Most apt/npm deps are baked into the per-task Dockerfile
        # already; warmup.sh inside the container is the fallback path for
        # interactive or post-image patches.
        meaningful = [
            line for line in warmup_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
            and line.strip() not in {"set -e"}
        ]
        if not meaningful:
            logger.debug(
                "wildclaw-v1: %s warmup.sh is header-only; skipping",
                task.get("task_name"),
            )
            return

        # Push warmup.sh into the container and exec it via the
        # runtime-agnostic SDK shell session — works on docker and modal
        # identically because both expose ``/v1/shell`` on the sandbox.
        dest = f"{self.container_workspace.rstrip('/')}/warmup.sh"
        if not runtime.copy_to_runtime(warmup_path, dest):
            raise RuntimeError("wildclaw-v1: failed to push warmup.sh into container")

        result = runtime.exec_in_runtime(
            f"bash {shlex.quote(dest)}",
            workdir=self.container_workspace,
            timeout=600,
        )
        if result.get("returncode", 0) != 0:
            logger.error(
                "wildclaw-v1: warmup.sh failed: %s",
                result.get("error") or result.get("output"),
            )

    # --- In-container, post-rollout --------------------------------------

    def inject_ground_truth(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any]
    ) -> None:
        gt_dir = task.get("gt_dir")
        if not gt_dir:
            logger.debug(
                "wildclaw-v1: %s has no gt/ — verifier may be LLM-judge based",
                task.get("task_name"),
            )
            return

        target = f"{self.container_workspace.rstrip('/')}/gt"
        gt_path = Path(gt_dir)
        for entry in sorted(gt_path.iterdir()):
            dest = f"{target}/{entry.name}"
            ok = runtime.copy_to_runtime(str(entry), dest)
            if not ok:
                raise RuntimeError(
                    f"wildclaw-v1: inject_ground_truth failed for {entry} → {dest}"
                )
        logger.info(
            "wildclaw-v1: injected %d gt entries into %s",
            sum(1 for _ in gt_path.iterdir()),
            target,
        )

    def score(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any],
        rollout_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        grade_path = task.get("grade_file_path")
        if not grade_path:
            logger.debug(
                "wildclaw-v1: %s has no grade.py — LLM-judge or skipped",
                task.get("task_name"),
            )
            return None

        # Push grade.py into the container alongside results/gt, then exec
        # it. We use the same copy_to_runtime API as setup_workspace so the
        # protocol stays runtime-agnostic.
        ws = self.container_workspace
        dest = f"{ws}/grade.py"
        if not runtime.copy_to_runtime(grade_path, dest):
            raise RuntimeError(f"wildclaw-v1: failed to push grade.py into container")

        # Inject ground truth last, so it's invisible during the rollout.
        self.inject_ground_truth(runtime, task)

        # Run grade.py in-container via the runtime-agnostic SDK shell.
        # We use python3 -c with a single-line program so stdout is one
        # JSON document we can parse straight off the wire — no need to
        # round-trip through a file inside the container.
        py = (
            f"import json,sys; sys.path.insert(0,{ws!r}); "
            "from grade import grade; print(json.dumps(grade()))"
        )
        # Pass env vars inline so the LLM-judge graders (39/60 tasks)
        # can read OPENROUTER_API_KEY etc. via os.environ. This is on
        # top of /etc/profile.d/uda_env.sh because exec_command may
        # spawn non-login shells that skip profile.d.
        result = runtime.exec_in_runtime(
            f"python3 -c {shlex.quote(py)}",
            workdir=ws,
            timeout=task.get("timeout_seconds", 600),
            env=self.get_extra_env(task),
        )
        if result.get("returncode", 0) != 0:
            err = result.get("error") or result.get("output") or "<no output>"
            logger.error("wildclaw-v1: grade.py failed: %s", err)
            return {"error": f"grade.py failed: {err}"}

        output = result.get("output", "")
        try:
            return json.loads(output.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            logger.error("wildclaw-v1: could not parse grade output: %s", exc)
            return {"error": f"unparseable grade output: {output!r}"}
