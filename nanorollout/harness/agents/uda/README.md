# UDA Agent Scaffold (Qwen + SGLang)

This scaffold drives **Qwen3.6 / Qwen3-VL via local SGLang inference**
through cocoa-v1 / wildclaw-v1 / osworld-v1 task rollouts. It is the loop
that turns a model + a sandbox into a verifiable RL-training trajectory
producer.

Anthropic / OpenAI / Gemini paths are kept as alternates (live in the
same `controller.py`), but the foreground design — system-prompt
brevity, context compaction, image-pruning intervals, retry policy — is
tuned for **SGLang prefix-cache economics**, not Anthropic
`cache_control` markers.

## 1. Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ TaskExecutor                            envs/uda_env/__init__.py    │
│   - per-iteration agent loop                                        │
│   - dispatches actions to SandboxClient / RuntimeAdapter            │
│   - asks Controller for the next action each turn                   │
│                                                                     │
│ ┌─ Controller (BaseLLM + provider subclasses)                       │
│ │   - QwenLLM (default — Qwen3.6 / Qwen3.5 / Qwen3-VL)              │
│ │   - OpenAILLM / ClaudeLLM / GeminiLLM / KimiLLM / GLMLLM / …      │
│ │   - owns self.messages, system_prompt, history compaction         │
│ │                                                                  │
│ │ ┌─ prompts.py    modular system-prompt builder                   │
│ │ ├─ history.py    image pruning + sliding-window compaction       │
│ │ └─ controller.py BaseLLM + provider subclasses                   │
│                                                                     │
│ UDAAgent (54 LOC)                       harness/agents/uda/uda_agent.py
│   - thin BaseAgent wrapper around TaskExecutor                      │
└─────────────────────────────────────────────────────────────────────┘
```

`UDAAgent.run_task(task)` is a 1-line forward to `TaskExecutor.run_task(task)` —
all interesting behaviour lives below.

## 2. The run loop (`TaskExecutor.run_task`)

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

Key invariants:

- **Stable prefix** — `controller.start_task` seeds `messages[0]` = `Task: …`;
  subsequent turns only append. The system prompt is set once on the
  controller and injected as the request-level first message (OpenAI
  spec) or `system=` param (Claude spec). Across multiple rollouts on
  the same task with the same model, the **entire prefix up to the
  first feedback turn is byte-identical** — exactly what SGLang's
  RadixAttention prefix cache rewards.
- **Single source of truth = `controller.messages`** — TaskExecutor no
  longer holds a `prompt` string; the dual-track from the pre-refactor
  scaffold is gone.
- **No `add_progress_note`** — the legacy "iteration N/MAX" appended to
  every user turn would mutate the prefix every call, defeating prefix
  cache. Removed entirely.
- **Screenshot-after-action is opt-in** (`sandbox.screenshot_after_action`,
  default `false`). For RL rollouts we want short trajectories; the
  model takes screenshots itself when it actually needs them.

## 3. System prompt (`prompts.py`)

Built once per task by `build_system_prompt(client_type, model_name=...)`
and stored on the controller as `self.system_prompt`. Style ported from
`anthropic-quickstarts/computer-use-best-practices`: a short stable core
+ per-tool-family addenda gated by `client_type`.

```
<core capability + universal guidelines>      _CORE
<computer_use addendum>  (if GUI in scope)    _COMPUTER_USE
<file addendum>          (if file tools)      _FILE
<code addendum>          (if code tools)      _CODE
<shell addendum>         (if shell tools)     _SHELL
<cross-tool workflow>    (if "unified")       _CROSS_TOOL
<Kimi relative-coordinate rule> (if Kimi)     _KIMI_RELATIVE_COORDINATES
```

Sizes (vs the legacy ~210-line `UNIFIED_INITIAL_PROMPT_TEMPLATE`):

| client_type | Lines | Chars | Use case |
|---|---|---|---|
| `unified` (default) | ~57 | 3.1 KB | UDAAgent default (GUI + file + code + shell) |
| `computer-use` | ~38 | 2.1 KB | GUI-only |
| `shell` / `file` | ~22 | 1.2 KB | CLI-only / file-only |

**Task instruction lives in the first user message**, not the system
block. This makes the system prefix shareable across tasks within a
`client_type` + model — both Anthropic prompt cache and SGLang
RadixAttention hit on the second call.

### Qwen3-VL is on the legacy path

Qwen3-VL parses `<tool_call>...</tool_call>` XML out of the user-visible
prompt rather than using API-level tool calling. `start_task` detects
`is_qwen_vl_model` (matches "qwen3-vl" or "qwen3_vl" in the model name)
and keeps the legacy `UNIFIED_INITIAL_PROMPT_TEMPLATE_QWEN3VL`
(instruction + tool descriptions all in one user message). For Qwen3-VL,
`system_prompt` stays empty.

Qwen3.6 / Qwen3.5+ match neither flag, so they take the normal
system/user split path with API-level tool calling (handled by SGLang's
`--tool-call-parser` server-side).

## 4. Tool defs (`envs/uda_env/tools.py`)

Tool schemas are **OpenAI function-call format**, registered by
`client_type`. Structure ported 1:1 from
[`anthropic-quickstarts/computer-use-best-practices/computer_use/tools/`](https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-best-practices/computer_use)
— a single `computer_use` tool with an `action` enum (vs the legacy
17-separate-tool layout), plus a `computer_batch` tool for executing
multiple actions in one turn.

| `client_type` | Tool set | Count |
|---|---|---|
| `unified` (default) | computer_use + computer_batch + editor + python + bash + task_complete | **6** |
| `computer-use` | computer_use + computer_batch + task_complete | 3 |
| `file` | editor + task_complete | 2 |
| `code` / `jupyter` | python + task_complete | 2 |
| `shell` | bash + task_complete | 2 |

Names align 1:1 with `anthropic-quickstarts/computer-use-best-practices/computer_use/tools/`:
`editor` (was `str_replace_editor` + 7 single-purpose file_* tools),
`python` (was `code_execute`, dropped `language`/`timeout` — Python-only),
`bash` (was `shell_execute`). `task_complete` is our addition — verifiers
hook into its `result` payload; best-practices uses `stop_reason="end_turn"`
instead, which doesn't fit our verifier model.

Legacy names (`shell_execute`, `code_execute`, `str_replace_editor`,
`file_read`, `file_write`, `file_list`, `replace_in_file`, `image_read`)
are still parseable in `map_tool_call_to_action` — they get remapped to
the new shape transparently, so fine-tuned models trained on the old
schema keep working. `search_in_file` / `find_files` are dropped with no
remap (use `bash` with `grep` / `find` instead).

`map_tool_call_to_action(tool_name, arguments)`:

- For `computer_use`: extracts `action` from arguments, validates remaining
  params against `_COMPUTER_USE_VALID_PARAMS[f"computer_use_{action}"]`,
  returns `{action_type: f"computer_use_{action}", ...cleaned_args}`.
- For `computer_batch`: expands `actions` list into `{actions: [...]}`
  which `TaskExecutor.run_task` already handles as a multi-action turn.
- For legacy names: no-op remap to the new shape so old fine-tunes
  still work.

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
At default uda-desktop resolution `1920×1080`, bottom-right is `(1919, 1079)`.

**For Qwen3.6 without vision**: the GUI tools work only if the model
supports image input. If you're rolling out cocoa-v1 / wildclaw-v1 /
osworld-v1 against a text-only Qwen, restrict `client_type` to `shell`
or `unified-no-gui` (TODO add this preset) — the GUI tools will be in
the schema but the model can't read screenshots.

### Batch tool

`computer_batch` carries an `actions: [...]` array; each entry has the
same shape as a single `computer_use` call. The dispatcher unwraps it
into TaskExecutor's existing multi-action path — the batch is executed
sequentially against the sandbox, stopping on first error. **All
coordinates inside a batch refer to the screenshot taken before the
batch.** Include a `screenshot` action at the end of the batch if the
batch is likely to change visible state you need to verify.

## 5. Context management — RL-rollout optimised

Five transforms run before each LLM API call (in
`BaseLLM._compact_history()`, called inside `BaseLLM.call()` between
user-message append and `_make_api_call`):

### (a) Image pruning — `history.StripImagesAtIntervals` (DEFAULT ON)

**This is the primary context-management mechanism for RL training,
and matters even more for SGLang RadixAttention than for Anthropic
ephemeral cache.**

Naive "keep last N images" replaces a *different* old image on every
turn past N, so the byte prefix of the request changes and any prefix
cache misses every call. The interval scheme lets the kept-count climb
from `min_images` to `min_images + interval - 1` then drop back, so for
`interval` consecutive turns the *same* oldest images map to the *same*
placeholder text and the cache prefix is stable.

Defaults: `min_images=3, interval=8` — same as the
anthropic-quickstarts/computer-use-best-practices reference.

Auto-detects both message shapes:

- OpenAI Chat-Completions (Qwen via SGLang): `image_url` blocks inside
  `role="user"` content.
- Anthropic-native (Claude): `image` blocks, including ones nested inside
  `tool_result` blocks.

Pruned images become `{type: "text", text: "[Image Omitted]"}` placeholders
so the model still sees that *something was there*, just not the bytes.
**Crucially: text content (instructions, tool results, assistant
reasoning) is NEVER pruned by this transform** — only image bytes
become placeholders. The full action history is preserved verbatim.

### (b) Sliding-window compaction — `history.SlidingWindowCompactor` (DEFAULT OFF)

**Opt-in.** Default `history_window = 0` (disabled). When enabled with
`history_window > 0`, drops middle iteration blocks, keeps:

- `messages[0]` (the original `Task: …` user message — never dropped)
- The last `history_window` iteration blocks (user → assistant → tool…
  triplets)

Iteration-aligned (not token-aligned) so OpenAI's
`{role:"tool", tool_call_id}` references always stay paired with their
parent `{role:"assistant", tool_calls:[...]}` — orphaning a tool message
is a 400 on both OpenAI-compatible and Anthropic APIs.

**Why off by default for RL training**: the policy conditions on the
full action history. Dropping middle iterations loses information that
the model may have already learned to depend on. For most rollouts (up
to ~50 iters), image pruning alone keeps tokens manageable because the
text portion of past tool results is small relative to image bytes.

Enable only when:
- Context length actually threatens to blow up (very long rollouts,
  e.g. 100+ iters with heavy tool output).
- You're not running training (eval / smoke tests where some info loss
  is acceptable).

Recommended starting value if you do turn it on: `history_window = 16`
(more lenient than 8) — keeps roughly the last quarter of a 50-iter
rollout.

### (c) Cache-control breakpoints — `_set_trailing_cache_control` (Claude only)

**Not applicable to Qwen / SGLang.** Ported from best-practices for the
ClaudeLLM path: 1 system breakpoint + 3 rolling trailing breakpoints.
Lives in `ClaudeLLM._compact_history()` as a Claude-specific override;
the base `_compact_history()` (which Qwen / OpenAI / Gemini all
inherit) does NOT call it.

SGLang has automatic prefix caching (RadixAttention) — it doesn't need
explicit breakpoints. Our sliding window + interval image-pruning is
what feeds it: every byte-identical prefix becomes a cache hit
transparently. **The economics flip from "API dollars" to "GPU time +
TTFT"** but the lever is the same: stabilise the prefix.

### (d) API-error retry — `_call_with_retry`

Ported from `anthropic-quickstarts/computer-use-best-practices/loop.py`.
Wraps `_make_api_call` with **exponential backoff + jitter** for errors
classified as recoverable by `_is_recoverable_api_error`:

| Error class | Retryable? |
|---|---|
| 4xx client error (400 / 401 / 403 / 404 / 422) | No — raise immediately |
| 429 rate limit | Yes |
| 5xx server error (e.g. SGLang OOM, transient backend fault) | Yes |
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
(replay, follow-up requests, trajectory analysis).

## 6. Token / cost / throughput tracking

Every `_make_api_call` parses `response.usage` and tracks:

- `total_input_tokens` / `total_uncached_input_tokens` / `total_cached_tokens`
- `total_output_tokens` / `total_reasoning_tokens`
- `total_cost` in USD (per `MODEL_PRICING_REGISTRY` in `controller.py`)
  — relevant for Anthropic/OpenAI/Google paths; for Qwen / SGLang the
  cost is GPU time, not API dollars, so this is informational only.

Logged per call and surfaced in the result payload as `api_cost_stats`
(see `TaskExecutor.run_task` tail).

**For RL training**: the metrics that matter are `total_input_tokens` +
`total_cached_tokens` ratio (cache hit rate proxy) and per-iter
`llm_call_s` (SGLang throughput). Both surface through `timing_stats` in
the result payload.

## 7. Provider-specific surface

All providers share `BaseLLM.call(prompt, images)` →
`_compact_history()` → provider-specific `_make_api_call()` →
`_handle_api_response()`. What differs:

| Provider | Default for our use? | System prompt | Tool schema | Message format |
|---|---|---|---|---|
| **QwenLLM (3.6 / 3.5+)** | ✅ — SGLang local | `{role:"system"}` prepended | OpenAI `tools=` (parsed server-side by SGLang `--tool-call-parser`) | OpenAI Chat-Completions + `extra_body.reasoning` |
| **QwenLLM (Qwen3-VL)** | ✅ — VL variant | empty (`""`) — tools embedded in user prompt | none (text-based `<tool_call>` parsing) | OpenAI Chat-Completions; no `tools=` |
| **OpenAILLM** | alternate | `{role:"system"}` prepended | OpenAI `tools=` | OpenAI Chat-Completions |
| **ClaudeLLM** | alternate | `system=` API param with `cache_control: ephemeral` | Anthropic `tools=` (converted from OpenAI shape) | Anthropic-native (`tool_use` / `tool_result` blocks) + `_set_trailing_cache_control` |
| **GeminiLLM** | alternate | `system_instruction` on `GenerateContentConfig` | Gemini `tools=[…]` (converted) | Gemini `contents=[…]` (converted) |
| **KimiLLM** / **GLMLLM** / **DeepSeekLLM** | alternate | inherits OpenAILLM | OpenAI `tools=` | OpenAI Chat-Completions |

## 8. Configuration

All knobs flow through `llm_config` (which the runner builds from CLI flags
in `_build_uda_config`):

| Knob | Default | What it does | Relevant for Qwen/SGLang? |
|---|---|---|---|
| `model` | (required) | Provider auto-detected from name | yes |
| `base_url` | (required) | SGLang OpenAI-compatible endpoint URL | yes — e.g. `http://127.0.0.1:8001/v1` |
| `temperature` | provider default | Sampling temperature | yes |
| `max_tokens` | provider default | Output budget | yes |
| `max_parse_retries` | `5` | Times to re-prompt on tool-call parse error | yes |
| `api_retry_max_attempts` | `5` | API-error retry attempt cap | yes — for SGLang OOM/disconnect |
| `api_retry_base_delay` | `1.0` | Base for exponential backoff (seconds) | yes |
| `history_window` | `0` (off) | Sliding-window K (iterations to retain). 0 = full history kept; >0 only when context truly bloats | rarely — keep off for RL |
| `image_prune_min` | `3` | Min images always kept | yes — vision tasks |
| `image_prune_interval` | `8` | Interval for cache-friendly image pruning | **yes — feeds SGLang prefix cache** |
| `sandbox.screenshot_after_action` | `false` | Force-screenshot after every GUI action (legacy) | yes — keep `false` for short trajectories |
| `sandbox.client_type` | `unified` | Picks the tool set + system-prompt sections | yes |

### Recommended Qwen3.6 + SGLang config

```python
llm_config = {
    "model": "Qwen/Qwen3.6-27B",
    "base_url": "http://127.0.0.1:8001/v1",
    "api_key": "EMPTY",                # SGLang ignores this
    "temperature": 0.0,                # deterministic for RL rollouts
    "max_tokens": 4096,
    # Context management:
    "image_prune_min": 3,              # default — image bytes pruned to placeholder
    "image_prune_interval": 8,         # default — cache-friendly rolling drop
    "history_window": 0,               # default — FULL action history kept
    # Retry:
    "max_parse_retries": 5,
    "api_retry_max_attempts": 5,
    "api_retry_base_delay": 1.0,
}
sandbox_config = {
    "client_type": "unified",
    "screenshot_after_action": False,
}
```

`history_window = 0` (the default) is the right call for RL training:
the policy needs the full action history to learn from. Image pruning
alone usually keeps tokens in check because the text portion of tool
results is small relative to image bytes. Bump `history_window` to
something like 16 only if you observe context overflow on
exceptionally long rollouts.

### SGLang server-side flags (relevant)

```
--enable-prefix-caching           # RadixAttention — enabled by default in recent SGLang
--tool-call-parser qwen3-coder    # parses Qwen3.6 OpenAI-style tool calls
--reasoning-parser qwen3          # captures Qwen3 reasoning_content into the right field
--mem-fraction-static 0.85        # leave room for KV cache growth across long rollouts
```

## 9. What was removed in the alignment refactor

- **`add_progress_note(prompt, iter)`** in `TaskExecutor` — its per-turn
  iteration counter invalidated SGLang prefix cache every call.
- **`controller.build_prompt(task_description=...)` /
  `build_prompt(feedback=...)` from the hot path** — replaced by
  `start_task` + `step`. The methods still exist on `BaseLLM` (used by
  Qwen3-VL internally), but `TaskExecutor` no longer calls them.
- **`_cleanup_old_user_message_images`** — replaced by
  `StripImagesAtIntervals`. The method is kept as a no-op shim so
  ClaudeLLM's override doesn't break if called externally.
- **Root-level `cache_control: {"type": "ephemeral"}` on the Claude
  request** — moved onto the system block + trailing user-content
  blocks. (Claude-only; Qwen path doesn't touch it.)
- **The 210-line `UNIFIED_INITIAL_PROMPT_TEMPLATE`** — kept in
  `controller.py` for the Qwen3-VL legacy path only; all other providers
  now read from `prompts.build_system_prompt()`.
- **17 separate `computer_use_*` tools** — collapsed into 1
  `computer_use` tool + action enum (+ `computer_batch`). Schema bytes
  ~5KB → ~1KB in the request prefix.
- **8 file_* tools** — collapsed into a single `editor` tool (5 commands).
- **`shell_execute` / `code_execute` names** — renamed to `bash` /
  `python` to match the canonical Anthropic / OpenHands naming.

## 10. Files

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

## 11. Cross-scaffold design lineage

Where each part came from, and why we made the choices we did:

| Concept | Origin | Why we kept / changed |
|---|---|---|
| 19-action `computer_use` + action enum | `anthropic-quickstarts/computer-use-best-practices` | Smaller schema, 1:1 with Anthropic Action_20251124, batch-friendly |
| `computer_batch` | best-practices | Matches Anthropic's recommended batch pattern |
| `editor` with 5 commands | best-practices + OpenHands (both identical here) | Replaces our legacy 8-tool file family |
| `bash` / `python` naming | best-practices + Claude Code's `Bash` | Industry-standard |
| Image-pruning interval scheme | best-practices `formatters.py` | Cache-friendly across both Anthropic and SGLang |
| `_call_with_retry` + `_is_recoverable_api_error` | best-practices `loop.py` | Same taxonomy works for SGLang (5xx = retry) |
| `_set_trailing_cache_control` | best-practices | Claude-only; Qwen path skips it |
| `[stopped before completing]` synthetic turn | best-practices | API validity for both protocols |
| Modular system-prompt sections | best-practices | Short, stable, per-`client_type` |
| Sliding-window compaction | our own (iteration-aligned) | OpenHands uses LLM-summarising condenser instead — we picked simpler |
| `task_complete` tool | our own | Best-practices uses `stop_reason == end_turn`; we kept the explicit tool because RL verifiers hook into `result` payload |
| Qwen3-VL legacy template | unchanged | F-tier non-goal in the design discussion: don't unify Qwen3-VL's text-prompt tool-call protocol |
| OpenHands' `task_tracker` / `delegate` / `apply_patch` / `grep` / `glob` | OpenHands | Not adopted — productivity-flavour, not RL-flavour |
| Claude Code's `Monitor` / `Cron*` / `PushNotification` / `EnterPlanMode` | Claude Code | Not adopted — these are UX/orchestration, not RL-relevant |
| Terminus 2's `commands[]` + `duration` | Terminus 2 | Not adopted — single-tool batched keystrokes conflicts with our `computer_batch` philosophy |
