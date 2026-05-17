# UDA Agent Scaffold

The UDA agent is the loop that turns a model + a sandbox into a task
executor. It lives in three layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│ TaskExecutor                            envs/uda_env/__init__.py    │
│   - owns the per-iteration loop                                     │
│   - dispatches actions to a SandboxClient / RuntimeAdapter          │
│   - asks the Controller for the next action each turn               │
│                                                                     │
│ ┌─ Controller (LLM-family abstract)    harness/agents/uda/         │
│ │   - OpenAILLM / ClaudeLLM / GeminiLLM / QwenLLM / KimiLLM / …    │
│ │   - owns self.messages, system_prompt, history compaction       │
│ │   - returns parsed action dicts                                  │
│ │                                                                  │
│ │ ┌─ prompts.py    modular system-prompt builder                   │
│ │ ├─ history.py    image pruning + sliding-window compaction       │
│ │ └─ controller.py BaseLLM + provider subclasses                   │
│                                                                     │
│ UDAAgent (54 LOC)                       harness/agents/uda/uda_agent.py │
│   - thin BaseAgent wrapper around TaskExecutor                      │
└─────────────────────────────────────────────────────────────────────┘
```

`UDAAgent.run_task(task)` is a 1-line forward to `TaskExecutor.run_task(task)` —
all interesting behaviour lives below.

## 1. The run loop (`TaskExecutor.run_task`)

```python
self.controller.start_task(task_desc)        # ① seed conversation
pending_feedback = None                       #
for iter in 1..max_iterations:                #
    if pending_feedback is None:              # ② first iter: replay seed
        action = controller._make_api_call(…) #    (no extra user msg)
    else:                                     # ③ rest: push feedback
        action = controller.step(             #
            pending_feedback,                 #
            images_base64=last_iter_images,   #
        )                                     #
    for sub in action["actions"]:             # ④ dispatch each tool call
        feedback = sandbox.get_feedback(sub)  #
        controller.add_tool_message(…)        #    role="tool" turn
        if screenshot_after_action and …:     # ⑤ optional post-action screenshot
            sandbox.take_screenshot()         #    OFF by default; agent picks
        if feedback["done"]: break            #
    last_iter_images = [last_screenshot] + image_reads
    pending_feedback = feedback["message"]    # ⑥ feed back into ②
```

Key differences from the pre-refactor scaffold:

- **No more dual-track prompt** — TaskExecutor doesn't hold a `prompt` string
  or call `controller.build_prompt(...)`. The single source of truth is
  `controller.messages` (seeded by `start_task`, extended by `step`).
- **No progress notes** — the old `add_progress_note(prompt, iter)` would
  append `[Progress update: iteration N/MAX. …]` to every user turn, which
  invalidated the Anthropic prompt cache every call. Removed.
- **Screenshot-after-action is opt-in** — set
  `sandbox.screenshot_after_action: true` in config to restore the old
  belt-and-suspenders behaviour. Default is off, matching the
  anthropic-quickstarts best-practices reference (the model takes
  screenshots when it needs them).

## 2. System prompt (`prompts.py`)

The system prompt is built once per task by `build_system_prompt(client_type,
model_name=...)` and lives on the controller as `self.system_prompt`.

It's deliberately short and modular, structured like
`anthropic-quickstarts/computer-use-best-practices/SYSTEM_PROMPT`:

```
<core capability + universal guidelines>      _CORE        ~18 lines
<computer_use addendum> (if GUI in scope)     _COMPUTER_USE
<file addendum>         (if file tools)       _FILE
<code addendum>         (if code tools)       _CODE
<shell addendum>        (if shell tools)      _SHELL
<cross-tool workflow>   (if "unified")        _CROSS_TOOL
<Kimi relative-coordinate rule> (if Kimi)     _KIMI_RELATIVE_COORDINATES
```

Sizes (vs the legacy ~210-line `UNIFIED_INITIAL_PROMPT_TEMPLATE` in
`controller.py`):

| client_type | Lines | Chars | Use case |
|---|---|---|---|
| `unified` | ~55 | 3.0 KB | UDAAgent default (computer-use + file + code + shell) |
| `computer-use` | ~38 | 2.1 KB | GUI-only |
| `shell` / `file` | ~22 | 1.2 KB | CLI-only / file-only |

**Task instruction lives in the first user message**, not the system block:
`controller.start_task(task)` pushes `{role: "user", content: "Task: …"}`.
This makes the system prefix shareable across tasks within a client_type
+ model, so Anthropic's prompt cache hits on the second call.

### Qwen3-VL is unchanged

Qwen3-VL parses `<tool_call>...</tool_call>` XML out of the user-visible
prompt and doesn't use API-level `tools=`. Migrating it to the
system/user split would change the inference contract, which the
optimization scope (item "F" in the design plan) explicitly excluded.
For Qwen3-VL, `start_task` keeps the legacy
`UNIFIED_INITIAL_PROMPT_TEMPLATE_QWEN3VL` (instruction + tool descriptions
all in one user message); `system_prompt` stays empty.

## 3. Tool defs (`envs/uda_env/tools.py`)

Tool schemas are **OpenAI function-call format**, registered by
`client_type`. Structure ported 1:1 from
[`anthropic-quickstarts/computer-use-best-practices`](https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-best-practices/computer_use)
— a single `computer_use` tool with an `action` enum (vs the legacy
17-separate-tool layout), plus a `computer_batch` tool for executing
multiple actions in one turn.

| `client_type` | Tool set | Count |
|---|---|---|
| `unified` (default) | computer_use + computer_batch + 8× file + code + shell + task_complete | **13** |
| `computer-use` | computer_use + computer_batch + task_complete | 3 |
| `file` | 8× file + task_complete | 9 |
| `code` / `jupyter` | code_execute + task_complete | 2 |
| `shell` | shell_execute + task_complete | 2 |

`map_tool_call_to_action(tool_name, arguments)` in `tools.py`:

- For `computer_use`: extracts `action` from arguments, validates remaining
  params against `_COMPUTER_USE_VALID_PARAMS[f"computer_use_{action}"]`,
  returns `{action_type: f"computer_use_{action}", ...cleaned_args}` so the
  downstream sandbox dispatcher (which strips the `computer_use_` prefix)
  is unchanged from the old 17-tool layout.
- For `computer_batch`: expands `actions` list into `{actions: [...]}` which
  `TaskExecutor.run_task` already handles as a multi-action turn.
- For legacy `computer_use_<action>` tool names: still accepted (no-op
  remap to the new shape) so fine-tuned models trained on the old schema
  don't break.
- For other tools (file_*, code_execute, shell_execute, task_complete,
  str_replace_editor): whitelisted in `_OTHER_TOOL_VALID_PARAMS`, arguments
  filtered to that whitelist, `action_type` = tool name verbatim.

### The 19 computer-use actions (Anthropic Action_20251124 + 2 ext.)

Single tool ``computer_use`` with an ``action`` enum. Schema mirrors
best-practices' [`ComputerTool.input_schema`](https://github.com/anthropics/claude-quickstarts/blob/main/computer-use-best-practices/computer_use/tools/computer.py).

| Action | Params used |
|---|---|
| `screenshot` | — |
| `cursor_position` | — |
| `mouse_move` | `coordinate` |
| `left_click` / `right_click` / `middle_click` / `double_click` / `triple_click` | `coordinate?`, `text?` (modifier keys held during click) |
| `left_click_drag` | `start_coordinate`, `coordinate`, `duration?` |
| `left_mouse_down` / `left_mouse_up` | `coordinate?` |
| `key` | `text` (xdotool keysym, e.g. `"ctrl+c"`, `"Return"`) |
| `type` | `text` (literal text into focused field) |
| `hold_key` | `text`, `duration` |
| `scroll` | `scroll_direction`, `scroll_amount`, `coordinate?`, `text?` (modifier) |
| `wait` | `duration` |
| `zoom` | `region=[x1,y1,x2,y2]` |
| `read_clipboard` | — (best-practices extension; uses xclip on uda-desktop) |
| `write_clipboard` | `text` (best-practices extension) |

Coordinates are integer pixels in the X11 root window, top-left origin.
At default resolution `1920×1080`, bottom-right is `(1919, 1079)`.

### Batch tool

`computer_batch` carries an `actions: [...]` array; each entry has the
same shape as a single `computer_use` call. The dispatcher unwraps it
into TaskExecutor's existing multi-action path — the batch is executed
sequentially against the sandbox, stopping on first error. **All
coordinates inside a batch refer to the screenshot taken before the
batch.** Include a `screenshot` action at the end of the batch if the
batch is likely to change visible state you need to verify.

### Tools are NOT described in the system prompt

Tool *descriptions* live in the OpenAI tool schemas themselves (each
tool has its own multi-line `description` field). The system prompt
only references tool names and points at where the schema lives. This
keeps the system block short and lets the model use the canonical
schema docs for parameter rules.

(Exception: Qwen3-VL has `{tools_description}` markdown-rendered into
its user-message prompt because it doesn't see API-level tool schemas.)

## 4. Context management

Three independent transforms run before each LLM API call, in
`BaseLLM._compact_history()` (called inside `BaseLLM.call()` between
user-message append and `_make_api_call`):

### (a) Sliding-window compaction — `history.SlidingWindowCompactor`

Drops oldest iteration blocks, keeps:

- `messages[0]` (the original `Task: …` user message — never dropped)
- The last `history_window` iteration blocks (user → assistant → tool…
  triplets)

Default `history_window = 8`. Configurable via `llm_config.history_window`.
Iteration-aligned (not token-aligned) so OpenAI's
`{role:"tool", tool_call_id}` references always stay paired with their
parent `{role:"assistant", tool_calls:[...]}` — orphaning a tool message
is a 400.

### (b) Image pruning — `history.StripImagesAtIntervals`

**This is the cache-friendly part.** Naive "keep last N images" replaces a
*different* old image on every turn past N, so the byte prefix of the
request changes and prompt cache misses every call. The interval scheme
lets the kept-count climb from `min_images` to `min_images + interval - 1`
then drop back, so for `interval` consecutive turns the *same* oldest
images map to the *same* placeholder text and the cache prefix is
stable.

Defaults: `min_images=3, interval=8` — same as the
anthropic-quickstarts/computer-use-best-practices reference.

Auto-detects both message shapes:

- OpenAI Chat-Completions: `image_url` blocks inside `role="user"` content.
- Anthropic-native (Claude): `image` blocks, including ones nested inside
  `tool_result` blocks.

Pruned images become `{type: "text", text: "[Image Omitted]"}` placeholders
so the model still sees that *something was there*, just not the bytes.

### (c) Cache-control breakpoints — `_set_trailing_cache_control` (Claude only)

Ported from `anthropic-quickstarts/computer-use-best-practices/loop.py`.
Anthropic allows ≤4 `cache_control` breakpoints per request:

1. **One on the system block** — set inside `ClaudeLLM._make_api_call`.
   Shared across all conversations with the same client_type+model.
2. **Up to 3 trailing on user-content blocks** — popped & re-applied each
   turn so the breakpoint set rolls forward with new turns instead of
   accumulating.

Together with (b), this means Claude's `cache_eff` (cache_read /
(cache_read + fresh_input)) stays high across long rollouts. Without (c),
the breakpoints are wrong; without (b), the cached prefix invalidates
every turn anyway.

### Token / cost tracking

Every `_make_api_call` parses `response.usage` and tracks:

- `total_input_tokens` / `total_uncached_input_tokens` / `total_cached_tokens`
- `total_output_tokens` / `total_reasoning_tokens`
- `total_cost` in USD (per `MODEL_PRICING_REGISTRY` in
  `controller.py`)

Logged per call and surfaced in the result payload as
`api_cost_stats` (see `TaskExecutor.run_task` tail).

### (d) API-error retry — `_call_with_retry`

Ported from `anthropic-quickstarts/computer-use-best-practices/loop.py`.
Wraps `_make_api_call` with **exponential backoff + jitter** for errors
classified as recoverable by `_is_recoverable_api_error`:

| Error class | Retryable? |
|---|---|
| 4xx client error (400 / 401 / 403 / 404 / 422) | No — raise immediately |
| 429 rate limit | Yes |
| 5xx server error | Yes |
| `APIConnectionError` / timeout / "overloaded" message | Yes |
| Anything else | Yes (default to retry to preserve prior behaviour) |

`delay = min(cap_delay, base_delay * 2**attempt) + uniform(0, 1)`.
Defaults: `api_retry_max_attempts=5`, `api_retry_base_delay=1.0`,
hard cap 60s. Configurable via `llm_config`.

Distinct from `max_parse_retries=5` which handles tool-call parse
errors (model emitted unparseable JSON) via correction-prompt retry
inside `_handle_api_response` — those don't pay backoff because they're
LLM-content errors, not transient API errors.

### (e) Synthetic assistant turn on loop exit

If `TaskExecutor.run_task` exits with `controller.messages[-1]` having
`role="user"` (max iterations reached on a tool-calling turn, parse
interrupted, model produced no tool calls), a synthetic
`{role: "assistant", content: "[stopped before completing]"}` is
appended so the conversation stays API-valid for downstream consumers
(replay, follow-up requests, trajectory analysis). Matches the
end-on-user handling in best-practices `loop.py:507-514`.

## 5. Provider-specific surface

All providers share `BaseLLM.call(prompt, images)` →
`_compact_history()` → provider-specific `_make_api_call()` →
`_handle_api_response()`. What differs per provider:

| Provider | System prompt | Tool schema | Message format |
|---|---|---|---|
| **OpenAILLM** | `{role:"system"}` prepended to `messages` | OpenAI `tools=[…]` | OpenAI Chat-Completions |
| **ClaudeLLM** | `system=` API param with `cache_control: ephemeral` | Anthropic `tools=[…]` (converted from OpenAI shape) | Anthropic-native (`tool_use` / `tool_result` blocks) |
| **GeminiLLM** | `system_instruction` on `GenerateContentConfig` | Gemini `tools=[…]` (converted) | Gemini `contents=[…]` (converted) |
| **QwenLLM** (3.5+) | `{role:"system"}` prepended | OpenAI `tools=` (server-side `--tool-call-parser`) | OpenAI Chat-Completions + `extra_body.reasoning` |
| **QwenLLM** (Qwen3-VL) | empty (`""`) — tools embedded in user prompt | none (text-based `<tool_call>` parsing) | OpenAI Chat-Completions; no `tools=` |
| **KimiLLM** | inherits OpenAILLM + Kimi rel-coord rule | OpenAI `tools=` | OpenAI Chat-Completions |
| **GLMLLM** / **DeepSeekLLM** | inherits OpenAILLM | OpenAI `tools=` | OpenAI Chat-Completions |

## 6. Configuration

All knobs flow through `llm_config` (which the runner builds from CLI flags
in `_build_uda_config`):

| Knob | Default | What it does |
|---|---|---|
| `model` | (required) | Provider auto-detected from name |
| `temperature` | provider default | Sampling temperature |
| `max_tokens` | provider default | Output budget |
| `max_parse_retries` | `5` | Times to re-prompt on tool-call parse error |
| `api_retry_max_attempts` | `5` | API-error retry attempt cap |
| `api_retry_base_delay` | `1.0` | Base for exponential backoff (seconds) |
| `history_window` | `8` | Sliding-window K (iterations to retain) |
| `image_prune_min` | `3` | Min images always kept |
| `image_prune_interval` | `8` | Interval for cache-friendly image pruning |
| `sandbox.screenshot_after_action` | `false` | Force-screenshot after every GUI action (legacy) |
| `sandbox.client_type` | `unified` | Picks the tool set + system-prompt sections |

## 7. What was removed in this refactor

- **`add_progress_note(prompt, iter)`** in `TaskExecutor` — its per-turn
  iteration counter invalidated prompt cache. The model now learns
  budget pressure via the iteration count appearing nowhere in the
  prompt; reviewers may decide to re-add it as a one-off final-iteration
  warning later if needed.
- **`controller.build_prompt(task_description=...)` /
  `build_prompt(feedback=...)` from the hot path** — replaced by
  `start_task` + `step`. The methods still exist on `BaseLLM` (used by
  Qwen3-VL internally), but `TaskExecutor` no longer calls them.
- **`_cleanup_old_user_message_images`** — replaced by
  `StripImagesAtIntervals`. The method is kept as a no-op shim so
  ClaudeLLM's override (which had a custom Anthropic-shape implementation)
  doesn't break if called externally.
- **Root-level `cache_control: {"type": "ephemeral"}` on the Claude API
  request** — moved onto the system block and onto the trailing 3
  user-content blocks. The original placement was a no-op (the Anthropic
  SDK does not accept `cache_control` at the request root).
- **The 210-line `UNIFIED_INITIAL_PROMPT_TEMPLATE`** — kept in
  `controller.py` for the Qwen3-VL legacy path only; all other providers
  now read from `prompts.build_system_prompt()`.

## 8. Files

```
harness/agents/uda/
├── README.md               (this file)
├── __init__.py             registry exports
├── base.py                 BaseAgent abstract
├── uda_agent.py            UDAAgent: thin wrap of TaskExecutor
├── controller.py           BaseLLM + 7 provider subclasses + legacy prompt templates
├── prompts.py              build_system_prompt() + modular sections
└── history.py              SlidingWindowCompactor + StripImagesAtIntervals (+ StripOldestImages)
```

Companion files outside this directory:

```
envs/uda_env/
├── __init__.py             TaskExecutor — owns the per-iteration loop
├── tools.py                OpenAI-format tool schemas (the canonical action docs)
├── base.py                 SandboxClient + ComputerUseSandboxClient + UnifiedSandboxClient
├── driver/                 per-bench task semantics (cocoa, wildclaw, osworld)
└── runtime_adapter/        per-runtime action translation (e.g. OSWorld pyautogui)
```
