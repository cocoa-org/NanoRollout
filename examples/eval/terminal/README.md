# Terminal Bench 2.0 Examples

Run a Terminal Bench 2.0 task with Terminus-2:

```bash
bash examples/eval/terminal/run_tb2_terminus2.sh
```

By default this runs `adaptive-rejection-sampler` and reads tasks from:

```text
https://github.com/harbor-framework/terminal-bench-2.git
```

The runner shallow-clones the repo into a local cache on first use, because the
task Docker build context and `tests/` directory need to exist on disk.

## Common Overrides

Run a different task:

```bash
INSTANCE_ID=build-cython-ext bash examples/eval/terminal/run_tb2_terminus2.sh
```

Run the predefined Terminal Bench 2.0 task set:

```bash
REQUEST_FILE=examples/eval/terminal/data/terminalbench_2.jsonl \
CONCURRENCY=4 \
bash examples/eval/terminal/run_tb2_terminus2.sh
```

Use a local Terminal Bench 2.0 checkout:

```bash
REPO_DIR=/path/to/terminal-bench-2 bash examples/eval/terminal/run_tb2_terminus2.sh
```

Run a specific repo revision:

```bash
REPO_REVISION=main bash examples/eval/terminal/run_tb2_terminus2.sh
```

Refresh the cached remote checkout before running:

```bash
REFRESH_REPO=true bash examples/eval/terminal/run_tb2_terminus2.sh
```

Use another model endpoint:

```bash
MODEL_NAME=anthropic/claude-opus-4-1 \
API_KEY="$ANTHROPIC_API_KEY" \
bash examples/eval/terminal/run_tb2_terminus2.sh
```

Override runner settings directly:

```bash
STEP_TIMEOUT=-1 \
EVAL_TIMEOUT=-1 \
ENV_TIMEOUT=120 \
CREATE_TIMEOUT=600 \
MAX_ITERATIONS=100 \
bash examples/eval/terminal/run_tb2_terminus2.sh
```

`step_timeout=-1` and `eval_timeout=-1` mean "use the per-task timeout from
the Terminal Bench task file."

Terminal-specific options are also normal CLI flags:

```bash
REPO_URL=https://github.com/harbor-framework/terminal-bench-2.git \
PARSER_NAME=json \
TIMEOUT_MULTIPLIER=1.0 \
bash examples/eval/terminal/run_tb2_terminus2.sh
```
