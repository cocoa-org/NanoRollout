# Brew evaluation server

This repo ships **one** helper script, [`launch_server_single.sh`](launch_server_single.sh), for starting the HTTP API on a single machine. For multinode Ray clusters usage, see [Multinode (manual Ray)](#multinode-manual-ray) below.

The API (`tbrew serve`) uses a Ray-backed scheduler (`brew.core.scheduler`).

## Prerequisites

- Run commands from the `brew` repo root using builtvirtual environment.
- Dependencies installed so `tbrew` is available (e.g. `uv sync` and `uv run`, or an activated venv with the package installed).

## Single node (script)

```bash
bash examples/server/launch_server_single.sh
```

Optional overrides (same `key=value` style as `tbrew`):

```bash
bash examples/server/launch_server_single.sh concurrency=64 port=12000
```

## Multinode (manual Ray)

Use this when workers should run on **other** hosts. Only the **head** machine runs `tbrew serve`; workers only run Ray.

### 1. Head node

Start Ray and note the address it prints (often `HOST:6379`).

```bash
ray start --head --dashboard-host=0.0.0.0
tbrew serve host=0.0.0.0 port=11000 concurrency=128
```

Clients send HTTP requests to this host (default port `11000` unless overridden).

### 2. Each worker node

Join the head using the address from step 1:

```bash
ray start --address='<HEAD_HOST:PORT>'
```

Leave that process running (or run Ray under your supervisor of choice). See [Ray cluster docs](https://docs.ray.io/en/latest/cluster/getting-started.html) for flags such as CPU/GPU counts.

### 3. Shutdown

On workers and head, when you are done:

```bash
ray stop --force
```

Stopping `tbrew` does not tear down Ray started via `ray start`; run `ray stop` explicitly if you want the cluster gone before the next launch.
