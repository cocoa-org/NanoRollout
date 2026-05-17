"""System-prompt builder for the UDA agent.

Modeled on anthropic-quickstarts/computer-use-best-practices:

* The system prompt is the **stable** part of the context: identity,
  environment, the few rules that matter. Tool *descriptions* live in
  the tool schemas themselves (see ``envs/uda_env/tools.py``), not in
  prose here. That keeps the system block short and cache-friendly.
* Per-tool-family addenda are conditionally appended by client_type so
  shell-only / file-only agents don't carry computer-use guidance.
* The task instruction is placed in the **first user message**, not the
  system prompt — so the system prefix is shared across tasks within a
  client_type and the Anthropic prompt cache can hit across runs.

Qwen3-VL still uses the legacy "everything in one user message" template
(controller.py:UNIFIED_INITIAL_PROMPT_TEMPLATE_QWEN3VL) because it
parses ``<tool_call>`` XML out of the user-visible prompt rather than
using API-level tool calling — F task in the optimization list said
"don't unify Qwen3-VL".
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Core capability block — short and stable. Mirrors
# anthropic-quickstarts/computer-use-best-practices SYSTEM_PROMPT.
# ---------------------------------------------------------------------------
_CORE = """\
You are a Universal Digital Agent operating an Ubuntu workstation (the
uda-desktop image). You see the screen via screenshots and act via the
provided tools.

Guidelines:
* Unless the task is purely textual, take a screenshot first to see the
  current state. Coordinates you emit refer to the most recent screenshot
  you were shown.
* Prefer programmatic interfaces over pixel control. For file I/O,
  downloads, computation, or anything with a CLI/API, use the shell, code,
  or file tools instead of clicking. Computer-use is for genuinely visual
  / interactive work that has no programmatic alternative.
* Before any GUI action, observe (screenshot). After a non-trivial GUI
  action, observe again to verify it took effect.
* If a click does nothing after one retry, use ``computer_use_zoom`` to
  re-read coordinates — you may be off by tens of pixels.
* When done, call ``task_complete``. If the task expects a specific
  output, pass it as ``result``. Do not fabricate URLs, filenames, or
  data; report gaps instead.
"""


_COMPUTER_USE = """\
Computer-use tools drive any GUI application via xdotool/scrot through
``/v1/computer-use/*``. There is no DOM and no selectors — coordinates are
**integer pixels in the most recent screenshot**, origin top-left, default
screen 1920x1080.

* ``computer_use`` — single tool with an ``action`` field. 19 actions
  (screenshot / left_click / double_click / triple_click / right_click /
  middle_click / mouse_move / left_click_drag / scroll / type / key /
  hold_key / left_mouse_down / left_mouse_up / cursor_position /
  read_clipboard / write_clipboard / wait / zoom). See the tool schema
  for per-action parameter rules.
* ``computer_batch`` — execute multiple ``computer_use`` actions
  sequentially in one turn. Stops on the first error. Coordinates inside
  a batch refer to the screenshot taken *before* the batch call. Prefer
  it over chained single calls whenever you can confidently predict two
  or more steps ahead; include a ``screenshot`` action at the end of
  the batch whenever the preceding actions are likely to change visible
  state you need to verify.
"""


_FILE = """\
Filesystem is rooted at ``/home/kasm-user/``. Use ``editor view <path>``
to inspect any file — for image extensions (.png/.jpg/.jpeg/.gif/.webp/.bmp)
it returns the file as a base64 image block; for text it returns numbered
lines. Never dump raw bytes / base64 / pixel arrays via ``python``.
"""


_CODE = """\
``python`` and ``bash`` both run at ``/home/kasm-user/``. Never print raw
image bytes / base64 / pixel arrays — only short metadata.
"""


_SHELL = _CODE  # combined guidance covers both; keep symbol for client_type=shell.


_CROSS_TOOL = """\
Cross-tool workflow:
* For downloads, computation, parsing, or anything with a CLI/API — prefer
  ``bash`` / ``python`` over clicking. Only fall back to a GUI download
  button when the URL is behind JS / session state a fetch can't replicate.
* Verify file outputs (existence, size, parseability) after writing.
* Stop after ~6 consecutive GUI actions without verified progress — drop
  to ``bash`` / ``python`` or call ``task_complete(result="failure_summary: …")``.
"""


# Kimi outputs coordinates in [0, 1] (relative). Server rescales to live
# resolution. Appended to system prompt only when the model is Kimi.
_KIMI_RELATIVE_COORDINATES = """\
Kimi coordinate rule: for ``coordinate`` and ``start_coordinate`` fields
on any computer-use action, emit **normalized values in [0, 1]** relative
to the current screenshot (``(0.5, 0.5)`` = screen center). Do not emit
absolute pixels.
"""


def _is_kimi(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    m = model_name.strip().lower()
    return "kimi" in m or "moonshot" in m


def build_system_prompt(
    client_type: str,
    *,
    model_name: Optional[str] = None,
) -> str:
    """Assemble the system prompt for a controller.

    Layout:

        <core capability + universal guidelines>
        <computer_use addendum (if client_type includes GUI)>
        <file addendum (if client_type includes file tools)>
        <code addendum (if client_type includes code tools)>
        <shell addendum (if client_type includes shell tools)>
        <cross-tool workflow (only for "unified")>
        <Kimi relative-coordinate rule (only for Kimi models)>

    Returns the assembled string (no trailing newline normalization —
    sections already include their own).
    """
    ct = (client_type or "").strip().lower()
    parts = [_CORE]

    if ct in ("unified", "computer_use", "computer-use", "browser"):
        parts.append(_COMPUTER_USE)
    if ct in ("unified", "file"):
        parts.append(_FILE)
    if ct in ("unified", "code", "jupyter"):
        parts.append(_CODE)
    if ct in ("unified", "shell"):
        parts.append(_SHELL)
    if ct == "unified":
        parts.append(_CROSS_TOOL)
    if _is_kimi(model_name) and ct in ("unified", "computer_use", "computer-use", "browser"):
        parts.append(_KIMI_RELATIVE_COORDINATES)

    return "\n".join(parts).strip() + "\n"
