# NanoRollout

Easy agent rollout at scale.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) with Python 3.12.

```bash
uv python pin 3.12
uv sync
```

This creates or reuses the project virtual environment and installs NanoRollout from
`pyproject.toml`/`uv.lock`.

If you prefer a minimal editable install instead of syncing the lockfile:

```bash
uv python pin 3.12
uv venv
uv pip install -e .
```

Check that the CLI is available:

```bash
nro --help
```

## CocoaBench

NanoRollout can run CocoaBench tasks natively through the in-repo Cocoa agent and
`nanorollout/envs/cocoa_env`. No separate `cocoa-agent` checkout is required for
the runtime path.

Example:

```bash
nro run \
  --task cocoa-bench \
  --agent cocoa-agent \
  --instance-id trader-joe-chip-shopping \
  --model-name gpt-5.2 \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --tasks-dir /path/to/cocoabench-example-tasks
```

Useful flags:

- `--cocoa-config`: overlay an existing CocoaAgent JSON config.
- `--controller-type`: force the CocoaAgent controller type instead of inferring it from the model name.
- `--client-type`: choose the Cocoa sandbox client type, default `unified`.
