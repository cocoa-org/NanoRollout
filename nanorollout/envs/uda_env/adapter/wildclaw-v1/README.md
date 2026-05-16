# wildclaw-v1 adapter

[InternLM/WildClawBench](https://github.com/InternLM/WildClawBench) — 60
hard end-to-end agent tasks across 6 categories, migrated to the UDA
adapter format so they run on the unified uda-desktop image through
`run_uda_agent`.

## Layout

Each task ships its schema in git; the input data (`exec/`) and ground
truth (`gt/`) come from HuggingFace at clone time.

```
adapter/wildclaw-v1/
├── README.md            (this file)
├── TASK_STATUS.md       per-task data-source classification (60 tasks)
├── .gitignore           excludes exec/ + gt/ — HF-sourced
├── _fetch_assets.sh     pulls exec/ + gt/ from HF for the 17 tasks that have them
└── <task_id>/
    ├── meta.json        {id, name, category, timeout_seconds, driver: "wildclaw-v1"}
    ├── Dockerfile       FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0 + per-task deps
    ├── docker-compose.yaml
    ├── task.yaml        instruction: |  (agent prompt, byte-identical to upstream)
    ├── grade.py         in-container verifier (byte-identical)
    ├── env.tsv          env vars to inject (e.g. OPENROUTER_API_KEY)
    ├── warmup.sh        in-container post-build setup
    ├── skills/          optional skill packages baked into image
    ├── exec/            (gitignored — HF-sourced) inputs at /tmp_workspace/
    └── gt/              (gitignored — HF-sourced) GT at /tmp_workspace/gt/
```

## Run one task

```bash
# 1. One-time: fetch HF assets for tasks that need them (17 of 60)
bash nanorollout/envs/uda_env/adapter/wildclaw-v1/_fetch_assets.sh

# 2. Run a task
INSTANCE_ID="01_Productivity_Flow_task_6_calendar_scheduling" \
BENCH="wildclaw-v1" \
MODEL_NAME="claude-sonnet-4-6" \
bash examples/eval/uda/run_uda_bench.sh
```

## What `run_uda_agent` does, per task

1. `load_driver_for_task_dir(task_dir)` → `WildclawV1Driver` (from
   `meta.json`'s `"driver"` field).
2. `driver.load_task(task_dir)` → reads `task.yaml`, paths to
   `grade.py / exec/ / gt/`, env.tsv list.
3. Container starts via docker-compose / modal sandbox.
4. `driver.setup_workspace(runtime, task)`:
   - stages env vars into `/etc/profile.d/uda_env.sh` + `/tmp_workspace/.env`
   - copies `exec/*` → `/tmp_workspace/*` via `runtime.copy_to_runtime`
5. `driver.run_warmup(runtime, task)` runs `warmup.sh` if non-trivial.
6. Agent rollout: standard `TaskExecutor.run_task` loop.
7. `driver.score(runtime, task, result)`:
   - pushes `grade.py` into container
   - injects `gt/` → `/tmp_workspace/gt/`
   - runs `python3 -c "from grade import grade; print(json.dumps(grade()))"`
     inside the container with env vars inlined
   - parses stdout JSON as the float-keyed score dict
8. Cleanup.

Both docker and modal runtimes go through the same code path because
`copy_to_runtime` and `exec_in_runtime` ride the agent-infra/sandbox
HTTP API exposed at port 8080 inside the container.

## Verifier semantics — float, multi-criterion

Unlike cocoa-v1's host-side pass/fail `test(result)`, wildclaw uses
in-container `grade() → dict[str, float]`. Score keys vary per task; all
60 tasks include `overall_score`. Examples:

| Task | Score dimensions |
|---|---|
| `01_Productivity_Flow_task_6_calendar_scheduling` | 17 keys: `output_files_valid`, `scheduled_ics_parseable`, `daily_limit_respected`, `optimality_ratio`, `overall_score`, ... |
| `01_Productivity_Flow_task_3_bibtex` | 17 keys including `arxiv_id_accuracy`, `title_accuracy`, `bibtex_exact_match_ratio`, `corrupted_paper_accuracy`, ... |

This preserves float-grained scoring for RL post-training.

## Migrator

Re-runnable from UDA-Gym:

```bash
python tools/migrate/wildclaw.py \
  --src   WildClawBench/tasks/<category> \
  --dst   NanoRollout/nanorollout/envs/uda_env/adapter/wildclaw-v1 \
  --skills-root WildClawBench/skills
```

## Status

See [TASK_STATUS.md](TASK_STATUS.md) for the per-task data-source
classification (17 with HF assets, 7 needing upstream `prepare.sh` for
videos / model weights, 14 fetching at rollout-time, 11 self-contained,
11 with no-source-yet input gaps).
