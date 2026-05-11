# NanoRollout

<p align="center">
  <img src="assets/nanorollout_logo.png" alt="NanoRollout" width="200"/>
</p>

<div align="center">Easy digital agent rollout at scale.</div>

<br>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://discord.gg/ywqJxCGc"><img src="https://img.shields.io/badge/Discord-join-7289da?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#agent-rl">Agent RL</a> |
  <a href="https://ember-factory-33d.notion.site/NanoRollout-A-Lightweight-Infra-for-Digital-Agent-Rollout-at-Scale-312927eea9bd803792f4c3b954f8daa1?pvs=74">Blog</a> |
  <a href="https://huggingface.co/cocoa-org">Huggingface</a> |
  <a href="https://wandb.ai/tinyagent-org/nanorollout_rl">Wandb</a>
</p>

## What is NanoRollout?
Scaling digital agents is bottlenecked by **environments**. Environments demand resources (CPU/memory) orthogonal to model training (GPU).
NanoRollout is a lightweight rollout repo that (1) **decouples** agent harnesses (e.g., OpenHands, mini-swe-agent, Terminus2, OSWorld-MM-Agent, Cocoa-Agent) and
environment backends (e.g., Docker, Modal, AWS EC2) from the trainer logic, so each can be developed and scaled independently; and (2) **unifies** the rollout service in evaluation, distillation, and reinforcment learning (RL) behind a single rollout server endpoint where clients submit a task and receive a trajectory.

Nanorollout powers fast parallel evaluation (SWE-Bench Verified in **18 min** with 500 workers), large-scale distillation (300K+ trajectories → *Mocha-Coder-32B*), and stable RL at large batch sizes (bsz 4,096 → *Mocha-RL-Alpha-32B*), integrating with [miles](https://github.com/radixark/miles), [veRL](https://github.com/verl-project/verl), and [tunix](https://github.com/google/tunix).


## Installation

```bash
git clone https://github.com/cocoa-org/NanoRollout.git
cd NanoRollout
```

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

For RL training, also fetch the `trainers/` submodule:

```bash
git submodule update --init --recursive
```

## Supported Agents

| Domain | Benchmark | Harness (`--agent`) | Sandbox (`--env-type`) |
|---|---|---|---|
| SWE | SWE-Bench Verified / Pro | `oh-core` (OpenHands), `oh-lite`, `mini-swe-agent`, `r2egym`, `claude-code`, `qwen-code`, `opencode` | `docker`, `modal`, `enroot` |
| Terminal | Terminal-Bench 2.0 | `terminus2`, `mini-swe-agent`, `claude-code`, `qwen-code`, `opencode` | `docker`, `modal`, `enroot` |
| Computer Use | OSWorld-Verified | `qwen3vl-mmagents` | `aws`, `docker`, et al. |
| Unified | CocoaBench | `cocoa-agent` | `docker`, `modal` |

## Quick Start

### `nro run` — Synchronous Rollout

Run a single SWE instance directly from the CLI:

```bash
nro run \
  --task swe --agent oh-core \
  --model-name deepseek-v4-flash \
  --base-url https://api.deepseek.com/v1 --api-key $OPENAI_API_KEY \
  --env-type docker --instance-id django__django-11095
```

Scale to 500 parallel workers on Modal:

```bash
nro run \
  --task swe --agent oh-core \
  --model-name deepseek-v4-flash \
  --base-url https://api.deepseek.com/v1 --api-key $OPENAI_API_KEY \
  --env-type modal \
  --request-file examples/eval/swe/data/swebench_verified.jsonl \
  --concurrency 500
```

`nro run` is best suited when environment resources are managed externally (e.g. Modal), so no Ray is needed. For self-hosted model endpoints (e.g. vLLM, SGLang), replace `--base-url` with your local endpoint (e.g. `--base-url http://<server-ip>:8000/v1`). For detailed examples across tasks (SWE-Bench, Terminal-Bench, OSWorld, CocoaBench) and agents, see [`examples/eval/`](examples/eval/).

### `nro serve` — Async Rollout Server

We recommend starting an async rollout server for flexible async requests and self-managed resources (like CPU/RAM), for evaluation, distillation, or RL training at scale.

```bash
ray start --head
nro serve host=0.0.0.0 port=11000 concurrency=64
```

Clients submit tasks to `POST /run` and receive trajectories with rewards and messages:

```bash
curl -s http://localhost:11000/run \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "django__django-11095",
    "task": "swe", "agent": "oh-core",
    "model_name": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "<your-api-key>"
  }'
```

RL trainers (miles, veRL, tunix) call this endpoint to generate rollout batches during training. See [`examples/server/`](examples/server/) for multi-node Ray cluster setup.

## Agent RL

NanoRollout serves trajectories to RL trainers through the same `POST /run` endpoint. Start `nro serve` (see [Quick Start](#quick-start)) first, then point your trainer at `NANOROLLOUT_URL=http://<host>:11000`. We have validated integration with [miles](https://github.com/cocoa-org/miles), [veRL](https://github.com/verl-project/verl), and [tunix](https://github.com/google/tunix); veRL and tunix reference code is coming soon.

### miles
The [miles](https://github.com/cocoa-org/miles) side captures exact tokens and logprobs from agent calls via a TITO proxy so the trainer sees the same token stream the agent saw. See [`miles/examples/nanorollout`](https://github.com/cocoa-org/miles/tree/main/examples/nanorollout) for the launch script, hyperparameters, and full setup for an example to train Qwen3-4B-Instruct.


