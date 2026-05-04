# Brew

Easy agent rollout at scale.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) with Python 3.12.

```bash
uv python pin 3.12
uv sync
```

This creates or reuses the project virtual environment and installs Brew from
`pyproject.toml`/`uv.lock`.

If you prefer a minimal editable install instead of syncing the lockfile:

```bash
uv python pin 3.12
uv venv
uv pip install -e .
```

Check that the CLI is available:

```bash
tbrew --help
```
