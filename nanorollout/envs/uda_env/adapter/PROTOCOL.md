# UDA bench-driver protocol

Every benchmark adapter under `adapter/<bench>/` plugs into the runner
via a single `BenchDriver` interface. The on-disk layout per task is
*by convention* similar across benchmarks (see
[`README.md`](README.md)), but the runtime behaviour (where ground
truth lives, where the verifier executes, how inputs reach the agent)
is encapsulated in the per-bench driver.

`TaskExecutor` and `run_uda_agent` are bench-agnostic: they look up the
driver via `meta.json`'s `driver` field and call into it. No
`if bench == "cocoa-v1" elif ...` ladders.

## The 7-method protocol

```python
class BenchDriver(Protocol):
    name: str                                       # "cocoa-v1", "wildclaw-v1"
    container_workspace: str                        # "/home/kasm-user", "/tmp_workspace"

    # ---- Offline / host-side ----
    def load_task(self, task_dir: Path) -> dict: ...
    def get_extra_env(self, task: dict) -> dict[str, str]: ...

    # ---- In-container, pre-rollout ----
    def setup_workspace(self, runtime, task): ...
    def run_warmup(self, runtime, task): ...

    # ---- In-container, post-rollout ----
    def inject_ground_truth(self, runtime, task): ...
    def score(self, runtime, task, rollout_result) -> dict | None: ...
```

`runtime` is a `BaseSandboxRuntime` (from `uda_env.base`) — drivers
interact with the container through two runtime-agnostic primitives,
**never** through `subprocess.run(["docker", ...])` directly:

| Primitive | Default impl | docker override | modal |
|---|---|---|---|
| `copy_to_runtime(host, container_path)` | walk + SDK `file.write_file` per file | native `docker cp` for speed | inherits default (HTTP → tunnel) |
| `exec_in_runtime(cmd, *, workdir, timeout)` | SDK `shell.create_session` + `shell.exec_command` | inherits default | inherits default |

Both rely on the agent-infra/sandbox HTTP surface (`/v1/file`, `/v1/shell`)
that lives inside every uda-desktop container. Docker exposes it on
`localhost:<HOST_PORT>`; modal exposes the same endpoint through a
`tunnel.url`. Drivers see one API, the runtimes route it.

## Lifecycle integration

`TaskExecutor` ([`uda_env/__init__.py`](../__init__.py)) calls into the
driver at three points:

1. **Runner-side, before container starts** —
   `run_uda_agent._load_task(task_dir)` now does
   `load_driver_for_task_dir(task_dir).load_task(task_dir)`. The
   returned dict has a `driver` field that flows through to every
   downstream lifecycle hook.

2. **In-container, after sandbox health-check, before agent runs** —
   `TaskExecutor.setup_environment` calls
   `driver.setup_workspace(runtime, task)` then
   `driver.run_warmup(runtime, task)`.

3. **After agent terminates** — `TaskExecutor.run_eval` calls
   `driver.score(runtime, task, result)`. The wildclaw driver also
   invokes `driver.inject_ground_truth(runtime, task)` internally from
   `score`, so GT is copied in last (not visible during rollout).

## Per-driver concretisation

### `cocoa-v1` ([`driver/cocoa_v1.py`](../driver/cocoa_v1.py))

- `container_workspace = "/home/kasm-user"`
- `load_task`: decrypt `task.yaml.enc` host-side with the canary, parse
  YAML, attach `test_file_path` pointing at `test.py.enc`.
- `get_extra_env`: pass through `task["env"]` if it's a dict.
- `setup_workspace`, `run_warmup`, `inject_ground_truth`: all no-ops —
  cocoa images already contain workspace + GT inside the canary-XOR'd
  `test.py.enc` closure.
- `score`: decrypt `test.py.enc` in-memory, exec it, call `test(result)`
  host-side. Returns whatever the verifier returns.

### `wildclaw-v1` ([`driver/wildclaw_v1.py`](../driver/wildclaw_v1.py))

- `container_workspace = "/tmp_workspace"`
- `load_task`: read plaintext `task.yaml` + `meta.json`. Records paths
  to `grade.py` / `exec/` / `gt/` / `warmup.sh` / `env.tsv` if present.
- `get_extra_env`: read names from `env.tsv`, look each up in the host
  `os.environ`, drop empties.
- `setup_workspace`: iterate over `exec/`, `runtime.copy_to_runtime`
  each entry into `/tmp_workspace/<name>`.
- `run_warmup`: skip if `warmup.sh` is header-only; otherwise push it
  in, then `runtime.exec_in_runtime("bash /tmp_workspace/warmup.sh", ...)`.
  Same code path on docker and modal.
- `inject_ground_truth`: iterate `gt/`, copy each into
  `/tmp_workspace/gt/<name>`. Only happens during `score`, so GT is
  invisible during rollout.
- `score`: push `grade.py` into the container, inject GT, then
  `runtime.exec_in_runtime("python3 -c '<inline>'", workdir="/tmp_workspace", ...)`.
  Parse the last stdout line as the float-keyed score dict.

## Adding a new driver

```python
# uda_env/driver/osworld_v2.py
from typing import TYPE_CHECKING
from .base import discover_workspace_assets
if TYPE_CHECKING:
    from ..base import BaseSandboxRuntime

class OSWorldV2Driver:
    name = "osworld-v2"
    container_workspace = "/tmp_workspace"

    def load_task(self, task_dir): ...
    # ...etc

# uda_env/driver/__init__.py
from .osworld_v2 import OSWorldV2Driver
register_driver(OSWorldV2Driver())
```

Then every task under `adapter/osworld-v2/<task_id>/meta.json` carries
`"driver": "osworld-v2"` and the rest of the system stays unchanged.

## Why this shape

- **Adapter authors think locally.** Each task ships exactly the assets
  the agent will see (`exec/`), the assets only the grader will see
  (`gt/`), and the grader itself — no cross-task coupling.
- **Hidden state stays hidden.** `gt/` arrives in-container only after
  the agent's rollout terminates, mirroring WildClawBench's "no data
  leakage during execution" guarantee. Cocoa achieves the same via
  encryption-with-canary on the verifier closure.
- **Runner is bench-agnostic.** `run_uda_agent` doesn't branch on
  benchmark name. New benchmark = new driver file + adapter content.
- **One verifier-execution model per bench, not per task.** Removes
  the per-task `if bench == "X" elif ... else` ladder that grows with
  every benchmark.
