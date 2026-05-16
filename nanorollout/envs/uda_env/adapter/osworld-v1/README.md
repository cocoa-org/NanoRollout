# osworld-v1 adapter

Drives the [OSWorld v1](https://github.com/xlang-ai/OSWorld) suite
(369 tasks across 10 domains: chrome, gimp, libreoffice_calc / impress /
writer, multi_apps, os, thunderbird, vlc, vs_code) using the **UDA
agent** instead of OSWorld's bundled VLM agent. The same 17
`computer_use_*` actions that drive cocoa / wildclaw on `uda-desktop`
also drive OSWorld on an EC2-backed AMI here — translated through
[`OSWorldV1Adapter`](../../runtime_adapter/osworld_v1.py).

## Why no per-task dirs?

Unlike cocoa-v1 / wildclaw-v1, OSWorld tasks ship as a single JSON
each (instruction + in-VM setup primitives + in-VM evaluator config),
and the bundled corpus already lives in this repo:

```
NanoRollout/examples/eval/osworld/data/
├── test_all.json                          # {domain: [task_id, …]} (369 entries)
└── examples/<domain>/<task_id>.json       # individual task configs
```

`OSWorldV1Driver.load_task()` reads directly from there; this directory
holds only docs.

## Architecture

```
UDAAgent  →  TaskExecutor  →  OSWorldV1Adapter  →  DesktopEnv (boto3)
   │                              │                      │
   │  computer_use_* dict          │  pyautogui code      │  POST {ec2_ip}:5000/execute
   │   {action_type, coord, …}     │  "pyautogui.click(x, y)"   (in-VM osworld-server)
   │                                                       │
   │                              evaluate()  ──────►  OSWorld evaluator
   │                                                   (float ∈ [0, 1])
```

`OSWorldV1Driver` (in [`driver/osworld_v1.py`](../../driver/osworld_v1.py))
is intentionally tiny: `setup_workspace` / `run_warmup` /
`inject_ground_truth` are no-ops because OSWorld's `env.reset()`
already runs its setup primitives (`launch`, `chrome_open_tabs`,
`googledrive`, …) and `env.evaluate()` self-contains the verifier.
Only `load_task` (read the OSWorld JSON) and `score` (call
`runtime.evaluate()`) carry weight.

## Action translation (17 + sentinels)

| UDA action_type | OSWorld VM dispatch |
|---|---|
| `computer_use_screenshot` | short-circuit: `GET /screenshot` → base64 |
| `computer_use_cursor_position` | short-circuit: `POST /run_python` with `pyautogui.position()`, parse stdout |
| `computer_use_mouse_move` | `pyautogui.moveTo(x, y)` |
| `computer_use_left_click` | `pyautogui.click(x, y)` (text= modifiers wrap with keyDown/Up) |
| `computer_use_right_click` | `pyautogui.rightClick(x, y)` |
| `computer_use_middle_click` | `pyautogui.middleClick(x, y)` |
| `computer_use_double_click` | `pyautogui.doubleClick(x, y)` |
| `computer_use_triple_click` | `pyautogui.tripleClick(x, y)` |
| `computer_use_left_click_drag` | `pyautogui.moveTo(sx,sy); pyautogui.dragTo(ex,ey, duration=…)` |
| `computer_use_left_mouse_down` | `pyautogui.mouseDown(button='left')` |
| `computer_use_left_mouse_up` | `pyautogui.mouseUp(button='left')` |
| `computer_use_key` | `pyautogui.hotkey('ctrl', 'c')` (xdotool→pyautogui keysym map) |
| `computer_use_type` | `pyautogui.typewrite(text, interval=0.01)` |
| `computer_use_hold_key` | `keyDown(k); sleep(d); keyUp(k)` |
| `computer_use_scroll` | `pyautogui.scroll(±amount)` / `hscroll(±amount)` |
| `computer_use_wait` | `env.step("WAIT", pause=duration)` |
| `computer_use_zoom` | short-circuit: `get_screenshot` + PIL crop region |
| `task_complete` / `exit` | `env.step("DONE")` + feedback.done=True |

### Coordinate space

The adapter maintains two sizes and a linear scaler between them:

- `agent_view_size` (default `(1920, 1080)`) — the viewport the model
  emits pixel coords in. Matches `tools.py:36`'s declared viewport for
  `uda-desktop`.
- `screen_size` (default `(1920, 1080)`) — the OSWorld VM's actual
  screen. Matches `DesktopEnv` defaults.

When the two match (the default), `CoordScaler.scale` is a no-op.
Override via `sandbox_config["agent_view_size"]` /
`sandbox_config["screen_size"]` if you fine-tune the agent on a
different viewport.

### Scroll units

Both Anthropic `scroll_amount` and `pyautogui.scroll(clicks)` are in
**mouse-wheel clicks** (one notch = one click). The adapter therefore
performs **no unit conversion**, only sign-flip by direction:

```
"up"    → +amount  (scroll)
"down"  → −amount  (scroll)
"right" → +amount  (hscroll, Linux-only)
"left"  → −amount  (hscroll, Linux-only)
```

`pyautogui.hscroll` only works on Linux; OSWorld VMs run Ubuntu, so
this is fine.

### xdotool → pyautogui keysyms

Anthropic Computer Tool uses xdotool keysyms (`Return`, `Page_Up`, `super`, …)
for `computer_use_key` and `computer_use_hold_key`. Most translate via
plain lowercase; the exceptions live in
[`XDOTOOL_TO_PYAUTOGUI`](../../runtime_adapter/base.py).

## Run one task

```bash
INSTANCE_ID=bb5e4c0d-f964-439c-97b6-bdb9747de3f4 \
BENCH=osworld-v1 \
ENV_TYPE=aws \
MODEL_NAME=claude-sonnet-4-6 \
bash examples/eval/uda/run_uda_bench.sh
```

`run_uda_bench.sh` already routes through the `--bench osworld-v1` code
path: `client_type` defaults to `"osworld-v1"`, `env_type` becomes the
OSWorld provider, and the runner resolves the task JSON inside
`examples/eval/osworld/data/`.

## Prerequisites

- AWS credentials in env: `AWS_REGION`, `AWS_SUBNET_ID`,
  `AWS_SECURITY_GROUP_ID`, plus standard AWS keys.
- `boto3` (already in the project's deps tree via `desktop_env`).
- A Claude / OpenAI / Qwen endpoint for the UDA controller.

## Not part of UDA-Gym training corpus

These 369 tasks are an **evaluation surface**, not training data. They
do not participate in the "verifiable cross-interface synthesis"
machinery (taxonomy / primitive composition / non-trivial floor) that
governs the synthesized UDA-Gym corpus.
