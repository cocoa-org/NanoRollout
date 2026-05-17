# UDA Examples

Run any migrated benchmark on the unified `uda-desktop` container image with
the UDA agent (computer-use + shell + file + code tools, no DOM browser layer).

The task corpus ships inside the package at
[`nanorollout/envs/uda_env/adapter/<bench>/`](../../../nanorollout/envs/uda_env/adapter/);
no external benchmark repo is needed at run time. Add a new benchmark by
dropping a migrated corpus under `adapter/<new-bench>/` and passing
`--bench <new-bench>` — no code changes in this directory required.

## Quickstart — CocoaBench v1.0 on UDA

```bash
bash examples/eval/uda/run_uda_bench.sh
```

Default: `BENCH=cocoa-v1`, `INSTANCE_ID=eight-puzzle-game`, `ENV_TYPE=modal`,
`MODEL_NAME=claude-sonnet-4-6`. Override any with env vars.

### Run a specific cocoa task

```bash
INSTANCE_ID=trader-joe-2 bash examples/eval/uda/run_uda_bench.sh
```

### Run all 153 v1.0 tasks at concurrency 50

```bash
CONCURRENCY=50 \
REQUEST_FILE=examples/eval/uda/data/cocoa_v1_all.jsonl \
bash examples/eval/uda/run_uda_bench.sh
```

### Local docker instead of Modal

```bash
ENV_TYPE=docker bash examples/eval/uda/run_uda_bench.sh
```

## How CocoaBench gets run **by uda-agent** in **uda_env**

The flow is intentionally short — one shared TaskExecutor / one shared
sandbox client / one shared agent surface, regardless of which benchmark
the task came from.

```
nro run --task uda --agent uda-agent --bench cocoa-v1 --instance-id eight-puzzle-game
  ↓
core/runners.py resolves -> harness/runner/uda/uda_agent:run_uda_agent
  ↓
run_uda_agent:
  - bench = "cocoa-v1"
  - task_dir = nanorollout/envs/uda_env/adapter/cocoa-v1/eight-puzzle-game/
  - decrypt task.yaml.enc + test.py.enc using canary.txt
  - build uda_config.json (controller + sandbox config, stamped with bench / image)
  ↓
UDAAgent(config):
  - executor = nanorollout.envs.uda_env.TaskExecutor(config)
  - controller from harness/agents/uda/controller.py:
      * loads tool defs from envs/uda_env/tools.get_unified_tools()
        (= 17 computer_use_* + file_* + code_execute + shell_execute + task_complete)
      * system prompt teaches computer-use vocabulary (no browser_*, no DOM, no BIDs)
  ↓
TaskExecutor loop:
  - setup_environment: Sandbox.create(image=task Dockerfile (FROM uda-desktop), encrypted_ports=[8080])
  - run_task: LLM step -> action_type:
       computer_use_*  -> ComputerUseSandboxClient -> POST /v1/computer-use/action (Anthropic ToolResult)
       shell_execute / file_* / code_execute  -> agent_sandbox SDK -> /v1/shell|file|code
       task_complete   -> exit loop
  - run_eval: exec test.py(.enc), call test(result) -> {"passed": bool, "feedback": str}
  - cleanup_environment: sandbox.terminate()
```

The same path serves any future benchmark: drop a migrated corpus at
`adapter/<name>/`, then `--bench <name>`.

## Adapter layout (current + planned)

```
nanorollout/envs/uda_env/adapter/
├── cocoa-v1/             ✅ 153 tasks (cocoabench-v1.0 migrated)
├── osworld-v2/           ⏳ planned — ~66% CLI-bypassable subset of OSWorld-Verified
└── swe-bench-mm/         ⏳ planned — 612 frontend bug fixes (npm test verifier)
```

OSWorld-V2 / SWE-bench-MM adapters need additional `_load_task` branches in
`run_uda_agent` (their verifier shape differs from cocoa's `test(result)`),
but the rollout loop / agent / env package are unchanged.

## Environment requirements

- `--env-type modal`: a `modal token set ...` profile with sandbox access.
- `--env-type docker`: local Docker daemon, plus `nanorollout/envs/uda_env/adapter/<bench>/<instance>/Dockerfile` resolves to a base image — the migrated cocoa-v1 corpus already points at `ghcr.io/bowenbryanwang/uda-desktop:0.6.0`.
- API key: `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) honored by the script.

## Override the docker base image benchmark-wide

The base is encoded in each task's `Dockerfile`. To swap globally, regenerate
the corpus with a different `--base-image`:

```bash
# from UDA-Gym workdir
python tools/migrate/cocoa.py \
    --src cocoa-agent/cocoabench-v1.0 \
    --base-image ghcr.io/bowenbryanwang/uda-desktop:0.7.0
```

## Override the runtime adapter root (e.g. for A/B testing migrations)

```bash
UDA_TASKS_DIR=/path/to/my/custom/adapter \
BENCH=cocoa-v1-experimental \
bash examples/eval/uda/run_uda_bench.sh
```
