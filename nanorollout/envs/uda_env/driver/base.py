"""Bench-driver protocol.

A `BenchDriver` encapsulates everything that differs *between benchmarks*
about how a task is loaded, how its workspace is staged inside the
sandbox container, how its ground truth is injected, and how its score
is computed.

Adding a new benchmark to UDA-Gym = write a new driver. The runner
(`run_uda_agent`) and the runtime (`TaskExecutor`) stay benchmark-agnostic:
they call into the driver via this protocol and never `if bench == ...`.

See ``adapter/PROTOCOL.md`` for the conceptual contract and per-driver
concretisation notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..base import BaseSandboxRuntime


@runtime_checkable
class BenchDriver(Protocol):
    """Per-benchmark adapter for the UDA env.

    All seven methods together cover the lifecycle:

    ::

        load_task            (offline, host-side)
        +-- get_extra_env    (just before container start)
        +-- container starts up via UnifiedSandboxClient
        +-- setup_workspace  (in-container, after start, before agent runs)
        +-- run_warmup       (in-container, optional)
        +-- agent.run_task   (uses instruction returned by load_task)
        +-- inject_ground_truth (in-container, after agent finishes)
        +-- score            (in-container or host-side, depending on driver)

    Implementations live in sibling modules and register themselves via
    ``register_driver`` in :mod:`nanorollout.envs.uda_env.driver`.
    """

    name: str
    container_workspace: str  # e.g. "/tmp_workspace" (wildclaw), "/home/kasm-user" (cocoa)

    # --- Offline / host-side ---------------------------------------------

    def load_task(self, task_dir: Path) -> Dict[str, Any]:
        """Read the task definition from disk.

        Must return a dict with at least:

        - ``task_dir`` (str)         — absolute path of the adapter dir
        - ``task_name`` (str)        — usually the dir's basename
        - ``instruction`` (str)      — agent-visible prompt
        - ``timeout_seconds`` (int)  — wall-clock budget
        - ``extra_env`` (dict)       — env vars to inject into container

        May add driver-specific keys (e.g. cocoa's ``test_file_path``,
        wildclaw's ``grade_file_path``); downstream code should never
        access those without first checking the driver name.
        """
        ...

    def get_extra_env(self, task: Dict[str, Any]) -> Dict[str, str]:
        """Env vars to pass into the container at start.

        Read from ``env.tsv`` (one var name per line) and mirrored from
        the host process environment. Empty values are dropped.
        """
        ...

    # --- In-container, pre-rollout ---------------------------------------

    def setup_workspace(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any],
    ) -> None:
        """Place input data into ``self.container_workspace`` inside the container.

        Drivers that ship inputs in the adapter directory (wildclaw's
        ``exec/``) copy them in here. Drivers that bake inputs into the
        image (cocoa) do nothing.

        Called once per rollout, after the container becomes healthy and
        before the agent issues its first action.
        """
        ...

    def run_warmup(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any],
    ) -> None:
        """Execute any post-start setup inside the container.

        wildclaw uses this for ``apt install / npm install -g`` lines
        that are too small to justify their own image layer.
        cocoa relies entirely on the image build, so this is a no-op.
        """
        ...

    # --- In-container, post-rollout --------------------------------------

    def inject_ground_truth(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any],
    ) -> None:
        """Place ground truth into the container after the agent stops.

        Crucial for "no data leakage during rollout" — the GT must not
        be visible until the agent has fully terminated. wildclaw copies
        ``gt/`` here; cocoa's GT lives inside the canary-XOR'd
        ``test.py.enc`` closure, so cocoa is a no-op here.
        """
        ...

    def score(
        self,
        runtime: "BaseSandboxRuntime",
        task: Dict[str, Any],
        rollout_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return per-criterion scores for this rollout.

        Implementation strategies differ widely:

        - **cocoa**: host-side. Decrypt ``test.py.enc`` with the canary,
          ``exec`` it, call ``test(rollout_result)`` which returns a
          bool or score dict.
        - **wildclaw**: in-container. Push ``grade.py`` into the sandbox
          via ``runtime.copy_to_runtime``, run ``python3 -c "from grade
          import grade; ..."`` via ``runtime.exec_in_runtime`` (which
          rides the agent-infra ``/v1/shell`` endpoint, runtime-agnostic
          across docker and modal), parse the JSON dict from stdout.

        May return ``None`` if no grader is bundled with the task (some
        wildclaw tasks rely on an external LLM judge configured by the
        adapter consumer).
        """
        ...


# Convenience subset --------------------------------------------------------

def discover_workspace_assets(task_dir: Path) -> Dict[str, Path]:
    """Return a dict of sub-paths present in ``task_dir``.

    Drivers use this to feature-detect optional inputs (``exec/``,
    ``gt/``, ``grade.py``, etc.) without each having to re-implement
    the same ``.is_dir() / .is_file()`` ladder.
    """
    candidates = {
        "exec": task_dir / "exec",
        "gt": task_dir / "gt",
        "skills": task_dir / "skills",
        "task_yaml": task_dir / "task.yaml",
        "task_yaml_enc": task_dir / "task.yaml.enc",
        "grade_py": task_dir / "grade.py",
        "grade_py_enc": task_dir / "grade.py.enc",
        "test_py": task_dir / "test.py",
        "test_py_enc": task_dir / "test.py.enc",
        "canary": task_dir / "canary.txt",
        "meta": task_dir / "meta.json",
        "env_tsv": task_dir / "env.tsv",
        "warmup_sh": task_dir / "warmup.sh",
    }
    return {k: v for k, v in candidates.items() if v.exists()}
