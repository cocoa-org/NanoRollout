"""LLM tool definitions for the UDA sandbox.

Tools are OpenAI-format function-call schemas. The agent picks from
these; the controller (``harness.agents.cocoa.controller``) emits a
``tool_call`` whose ``name`` becomes the ``action_type`` consumed by
``UnifiedSandboxClient.get_feedback``.

UDA replaces cocoa_env's ``browser_*`` / ``dom_*`` family (CDP-based,
DOM-aware, web-only) with a ``computer_use_*`` family (xdotool/scrot,
pixel-level, works on any X11 application). This matches the
``/v1/computer-use/*`` endpoint that ``uda-desktop`` exposes and
follows the Anthropic Computer Tool action vocabulary
(Action_20251124).

``file_*``, ``code_*``, ``shell_*`` families are byte-identical to
cocoa_env because the ``/v1/file/*`` / ``/v1/code/*`` / ``/v1/shell/*``
endpoints are inherited unchanged from agent-infra/sandbox.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# computer-use tools (replace cocoa_env's browser_* / dom_*)
# ---------------------------------------------------------------------------
#
# Action vocabulary follows Anthropic's Computer Tool Action_20251124 and is
# served by uda-desktop at POST /v1/computer-use/action. Each tool's name
# (e.g. ``computer_use_left_click``) is used directly as the ``action_type``;
# at dispatch time ``ComputerUseSandboxClient`` strips the ``computer_use_``
# prefix and forwards the remaining ``{action: 'left_click', ...}`` to the
# server.
#
# Coordinate semantics: integer pixel positions in the X11 root window,
# (0, 0) = top-left. Sizes match the kasm-user XFCE session resolution
# (default 1920x1080; see /v1/computer-use/health for the live value).

_COMPUTER_USE_DESCRIPTION = (
    "GUI control via uda-desktop's /v1/computer-use/* surface (xdotool + scrot). "
    "Use this for any visual / pixel-level interaction — desktop apps, web UIs, "
    "system dialogs. Most actions return a screenshot in base64_image."
)


# Canonical action set ported from
# anthropic-quickstarts/computer-use-best-practices/computer_use/tools/computer.py.
# Same 17 Anthropic Action_20251124 actions + read_clipboard / write_clipboard.
_COMPUTER_USE_ACTION_LIST: List[str] = [
    "screenshot",
    "left_click",
    "double_click",
    "triple_click",
    "right_click",
    "middle_click",
    "mouse_move",
    "left_click_drag",
    "scroll",
    "type",
    "key",
    "hold_key",
    "left_mouse_down",
    "left_mouse_up",
    "cursor_position",
    "read_clipboard",
    "write_clipboard",
    "wait",
    "zoom",
]


_COMPUTER_USE_ACTION_DESCRIPTION = (
    "* screenshot: capture the screen.\n"
    "* left_click / double_click / triple_click / right_click / middle_click: "
    "click at `coordinate`; optional `text` holds modifier keys "
    "(e.g. 'ctrl', 'shift') during the click.\n"
    "* mouse_move: move cursor to `coordinate`.\n"
    "* left_click_drag: drag from `start_coordinate` to `coordinate`.\n"
    "* scroll: scroll at `coordinate` in `scroll_direction` by `scroll_amount` notches; "
    "optional `text` holds modifier keys (e.g. 'ctrl' for zoom).\n"
    "* type: type literal `text` at the current focus.\n"
    "* key: press a chord like 'ctrl+shift+t' or a single key (xdotool keysym).\n"
    "* hold_key: hold the chord in `text` for `duration` seconds.\n"
    "* left_mouse_down / left_mouse_up: press or release the left button at "
    "`coordinate` (for manual drags or long-press).\n"
    "* cursor_position: return the current cursor x,y.\n"
    "* read_clipboard / write_clipboard: get/set clipboard `text`.\n"
    "* wait: sleep `duration` seconds.\n"
    "* zoom: return a cropped, higher-detail view of the screen region "
    "`region` = [x1, y1, x2, y2]. Use this to read small text or inspect fine "
    "detail. Coordinates in subsequent actions still refer to the full "
    "screenshot, not the zoom."
)


# Single-source input schema for the unified ``computer_use`` tool.
# Mirrors the JSON Schema in
# anthropic-quickstarts/computer-use-best-practices/computer_use/tools/computer.py.
_COMPUTER_USE_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": list(_COMPUTER_USE_ACTION_LIST),
            "description": _COMPUTER_USE_ACTION_DESCRIPTION,
        },
        "coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "start_coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "text": {"type": "string"},
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"],
        },
        "scroll_amount": {"type": "integer", "minimum": 1},
        "duration": {"type": "number", "minimum": 0, "maximum": 60},
        "region": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x1, y1, x2, y2] in the same image space as `coordinate`.",
        },
    },
    "required": ["action"],
}


def get_computer_use_tools() -> List[Dict[str, Any]]:
    """Get OpenAI tool definitions for the computer-use action set.

    Returns:
        A list of three tools:

        * ``computer_use`` — single tool with an ``action`` enum (19 actions)
          mirroring ``anthropic-quickstarts/computer-use-best-practices``
          (the canonical Anthropic Action_20251124 set + ``read_clipboard``
          / ``write_clipboard``). Drives uda-desktop's ``/v1/computer-use/*``.
        * ``computer_batch`` — execute multiple ``computer_use`` actions
          sequentially in one model turn. Stops on the first error.
          Coordinates inside a batch refer to the screenshot taken *before*
          the batch call.
        * ``task_complete`` — terminator sentinel; verifier hooks in via
          its ``result`` payload.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": _COMPUTER_USE_DESCRIPTION,
                "parameters": _COMPUTER_USE_INPUT_SCHEMA,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_batch",
                "description": (
                    "Execute multiple `computer_use` actions sequentially in a "
                    "single turn. Stops on the first error. Each item is the "
                    "same shape as a single `computer_use` call. All "
                    "coordinates refer to the screenshot taken *before* this "
                    "batch. Include a `screenshot` action at the end whenever "
                    "the preceding actions are likely to change visible state "
                    "you need to verify."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "minItems": 1,
                            "items": _COMPUTER_USE_INPUT_SCHEMA,
                        }
                    },
                    "required": ["actions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": (
                    "Mark the task as complete and exit. Optionally provide "
                    "the final result/answer if the task requires returning a "
                    "specific output (e.g., a JSON string). For tasks that "
                    "generate files in the sandbox, omit the `result` parameter."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": (
                                "Optional: the final result for the task. Use "
                                "when the task requires returning a specific "
                                "output. Omit when the task writes files."
                            ),
                        }
                    },
                },
            },
        },
    ]


def get_file_tools() -> List[Dict[str, Any]]:
    """File-side tools — a single ``editor`` modeled on best-practices.

    Replaces the legacy 8-tool file family (file_read/write/list,
    replace_in_file, search_in_file, find_files, image_read,
    str_replace_editor) with a single ``editor`` tool whose ``command``
    enum covers view/create/str_replace/insert (+ ``undo_edit`` for the
    in-sandbox SDK; best-practices drops it but we keep it because
    agent-infra exposes it). Schema mirrors
    anthropic-quickstarts/computer-use-best-practices/computer_use/tools/editor.py.

    ``editor view`` on an image file (.png/.jpg/.jpeg/.gif/.webp/.bmp)
    returns the file as a base64 image — the model sees the pixels.
    Lists / greps / finds: use ``bash`` instead.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "editor",
                "description": (
                    "View, create, and edit files inside the sandbox at "
                    "/home/kasm-user/. ``view`` shows a file (text with line "
                    "numbers, image as a base64 image, directory as a "
                    "listing). ``create`` writes a new file with file_text. "
                    "``str_replace`` swaps a unique occurrence of old_str for "
                    "new_str. ``insert`` inserts new_str after a line. "
                    "``undo_edit`` reverts the most recent edit."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                            "description": (
                                "* view: show a file (text+line numbers; "
                                "image returned as base64 if extension is "
                                ".png/.jpg/.jpeg/.gif/.webp/.bmp) or list a "
                                "directory.\n"
                                "* create: create/overwrite a file with `file_text`.\n"
                                "* str_replace: replace the unique occurrence "
                                "of `old_str` with `new_str` in `path`. Fails "
                                "if `old_str` is missing or not unique.\n"
                                "* insert: insert `new_str` after line "
                                "`insert_line` (0 inserts at the top).\n"
                                "* undo_edit: undo the most recent edit on "
                                "this file."
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Path relative to /home/kasm-user/ or an "
                                "absolute path under that root."
                            ),
                        },
                        "file_text": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer", "minimum": 0},
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "[start, end] 1-indexed; -1 for end means EOF.",
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        },
    ]


def get_code_tools() -> List[Dict[str, Any]]:
    """Code execution — a single ``python`` tool, Python-only.

    Replaces the legacy ``code_execute(code, language?, timeout?)`` which
    accepted ``language`` (always python in practice) and an unused
    ``timeout``. Schema mirrors best-practices PythonTool.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": (
                    "Run a Python 3 script inside the sandbox runtime. "
                    "Working directory is /home/kasm-user/. Returns "
                    "stdout/stderr. Never print raw image bytes / base64 / "
                    "pixel arrays — only short metadata (size, dimensions, "
                    "format)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        },
    ]


def get_shell_tools() -> List[Dict[str, Any]]:
    """Shell execution — a single ``bash`` tool.

    Replaces the legacy ``shell_execute`` (functionally identical, just
    aligned in name with best-practices BashTool).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    "Run a bash command inside the sandbox. Working directory "
                    "is /home/kasm-user/. Returns stdout/stderr. Prefer "
                    "wget/curl over a GUI download button, and grep/find over "
                    "screenshotting an editor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]


def get_unified_tools() -> List[Dict[str, Any]]:
    """Get unified OpenAI tool definitions combining all UDA sandbox capabilities.

    Returns:
        List of tool definitions combining computer-use, file, code, and shell tools.
        Browser (cocoa_env's CDP-based DOM tools) is intentionally NOT exposed —
        UDA agents reach GUIs through computer_use_* instead.
    """
    computer_use_tools = get_computer_use_tools()
    file_tools = get_file_tools()
    code_tools = get_code_tools()
    shell_tools = get_shell_tools()

    all_tools: List[Dict[str, Any]] = []
    task_complete_added = False

    for tool_set in [computer_use_tools, file_tools, code_tools, shell_tools]:
        for tool in tool_set:
            tool_name = tool["function"]["name"]
            if tool_name == "task_complete":
                if not task_complete_added:
                    all_tools.append(tool)
                    task_complete_added = True
            else:
                all_tools.append(tool)

    return all_tools


# Valid parameter sets per computer-use action.
# Keyed by the synthesized ``action_type`` (``"computer_use_<action>"``) that
# :func:`map_tool_call_to_action` produces, NOT by tool name (the tool itself
# is the single ``computer_use``). Used by sandbox-side validation to catch
# obviously wrong payloads before they hit ``/v1/computer-use/action``.
_COMPUTER_USE_VALID_PARAMS: Dict[str, set] = {
    "computer_use_screenshot": set(),
    "computer_use_cursor_position": set(),
    "computer_use_mouse_move": {"coordinate"},
    "computer_use_left_click": {"coordinate", "text"},
    "computer_use_right_click": {"coordinate", "text"},
    "computer_use_middle_click": {"coordinate", "text"},
    "computer_use_double_click": {"coordinate", "text"},
    "computer_use_triple_click": {"coordinate", "text"},
    "computer_use_left_click_drag": {"start_coordinate", "coordinate", "duration"},
    "computer_use_left_mouse_down": {"coordinate"},
    "computer_use_left_mouse_up": {"coordinate"},
    "computer_use_key": {"text"},
    "computer_use_type": {"text"},
    "computer_use_hold_key": {"text", "duration"},
    "computer_use_scroll": {"scroll_direction", "scroll_amount", "coordinate", "text"},
    "computer_use_wait": {"duration"},
    "computer_use_zoom": {"region"},
    "computer_use_read_clipboard": set(),
    "computer_use_write_clipboard": {"text"},
}

_OTHER_TOOL_VALID_PARAMS: Dict[str, set] = {
    # Aligned with anthropic-quickstarts/computer-use-best-practices.
    "editor": {"command", "path", "file_text", "old_str", "new_str", "insert_line", "view_range"},
    "python": {"code"},
    "bash": {"command"},
    "task_complete": {"result"},
}

# Legacy → new aliases. Old tool names are quietly remapped so that any
# fine-tuned model still emitting them keeps working. The current
# ``get_*_tools()`` only expose the new names, so over time these aliases
# stop firing in practice.
_LEGACY_TOOL_ALIASES: Dict[str, str] = {
    "shell_execute": "bash",
    "code_execute": "python",
    "str_replace_editor": "editor",
    # The 7 single-purpose file_* tools all collapse onto editor commands:
    "file_read": "editor",
    "file_write": "editor",
    "file_list": "editor",
    "replace_in_file": "editor",
    "image_read": "editor",
}

# Legacy file_* arguments → editor argument shape. ``command`` is filled
# in below by the dispatcher.
_LEGACY_FILE_TOOL_TO_EDITOR_COMMAND: Dict[str, str] = {
    "file_read": "view",
    "file_write": "create",
    "file_list": "view",
    "replace_in_file": "str_replace",
    "image_read": "view",
    "str_replace_editor": "",  # already shaped like editor; passes through
}


def _validate_computer_use_args(action: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unknown params for a given ``action``; raise on unknown actions."""
    action_type = f"computer_use_{action}"
    if action_type not in _COMPUTER_USE_VALID_PARAMS:
        raise ValueError(
            f"Unknown computer_use action: {action!r}. "
            f"Valid actions: {sorted(a for a in _COMPUTER_USE_VALID_PARAMS)}."
        )
    valid_params = _COMPUTER_USE_VALID_PARAMS[action_type]
    cleaned = {k: v for k, v in arguments.items() if k in valid_params}
    return cleaned


def map_tool_call_to_action(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Map an LLM tool call to a sandbox action.

    Dispatching rules:

    * ``computer_use`` — unified tool with an ``action`` enum (19 actions).
      Returns ``{"action_type": "computer_use_<action>", ...other_args}`` so
      downstream dispatch in :class:`ComputerUseSandboxClient` /
      :class:`OSWorldV1Adapter` (which both strip the ``computer_use_``
      prefix) keeps working unchanged.

    * ``computer_batch`` — wraps a list of sub-actions. Returns
      ``{"actions": [...]}`` which :class:`TaskExecutor.run_task` already
      handles as a multi-action turn.

    * Legacy ``computer_use_<action>`` names are still accepted (no-op
      remap to the new shape) so fine-tuned models trained on the old
      schema don't break.

    * Any other tool (file_*, code_execute, shell_execute, task_complete,
      str_replace_editor, …) is whitelisted in ``_OTHER_TOOL_VALID_PARAMS``;
      its arguments are filtered to that whitelist and ``action_type`` is
      set to the tool name verbatim.

    Raises ``ValueError`` for unknown tools / unknown actions / unsupported
    arguments.
    """
    # --- computer_batch: expand to multi-action shape ------------------- #
    if tool_name == "computer_batch":
        raw_actions = arguments.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError(
                "computer_batch requires a non-empty 'actions' array; "
                f"got {raw_actions!r}"
            )
        sub_actions: List[Dict[str, Any]] = []
        for i, sub in enumerate(raw_actions):
            if not isinstance(sub, dict):
                raise ValueError(f"computer_batch.actions[{i}] must be an object")
            sub_actions.append(map_tool_call_to_action("computer_use", sub))
        return {"action_type": "computer_batch", "actions": sub_actions}

    # --- computer_use (single action) ----------------------------------- #
    if tool_name == "computer_use":
        action_name = arguments.get("action")
        if not isinstance(action_name, str) or not action_name:
            raise ValueError("computer_use requires an 'action' field")
        rest = {k: v for k, v in arguments.items() if k != "action"}
        cleaned = _validate_computer_use_args(action_name, rest)
        return {"action_type": f"computer_use_{action_name}", **cleaned}

    # --- Legacy computer_use_<action> tool names ------------------------ #
    if tool_name.startswith("computer_use_") and tool_name in _COMPUTER_USE_VALID_PARAMS:
        action_name = tool_name[len("computer_use_"):]
        cleaned = _validate_computer_use_args(action_name, arguments)
        return {"action_type": tool_name, **cleaned}

    # --- Legacy aliases (best-practices alignment) ---------------------- #
    if tool_name in _LEGACY_TOOL_ALIASES:
        new_name = _LEGACY_TOOL_ALIASES[tool_name]
        if new_name == "editor":
            arguments = _normalize_legacy_file_args(tool_name, arguments)
        tool_name = new_name

    # --- bash / python / editor / task_complete -------------------------- #
    if tool_name not in _OTHER_TOOL_VALID_PARAMS:
        raise ValueError(f"Unknown tool: {tool_name}")
    valid_params = _OTHER_TOOL_VALID_PARAMS[tool_name]
    invalid_params = set(arguments.keys()) - valid_params
    if invalid_params:
        raise ValueError(
            f"Tool '{tool_name}' does not support parameters: {invalid_params}. "
            f"Valid parameters are: {valid_params}. "
            f"Received: {list(arguments.keys())}"
        )
    cleaned = {k: v for k, v in arguments.items() if k in valid_params}
    return {"action_type": tool_name, **cleaned}


def _normalize_legacy_file_args(legacy_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a legacy file_*-tool argument bag into the new ``editor`` shape.

    Handles the rename: file_read/file_list/image_read → editor view,
    file_write → editor create, replace_in_file → editor str_replace.
    Field renames (``old_text``→``old_str``, ``new_text``→``new_str``,
    ``file``→``path``) happen here so the rest of the pipeline only ever
    sees ``editor`` shape.
    """
    command = _LEGACY_FILE_TOOL_TO_EDITOR_COMMAND.get(legacy_name, "")
    if not command:
        # str_replace_editor: already in the new shape, just rename.
        return arguments

    out: Dict[str, Any] = {"command": command}
    # path: ``path`` (most) or ``file`` (replace_in_file).
    path = arguments.get("path") or arguments.get("file")
    if path is not None:
        out["path"] = path

    if legacy_name == "file_write":
        if "content" in arguments:
            out["file_text"] = arguments["content"]
    elif legacy_name == "replace_in_file":
        old = arguments.get("old_str") or arguments.get("old_text")
        new = arguments.get("new_str") or arguments.get("new_text")
        if old is not None:
            out["old_str"] = old
        if new is not None:
            out["new_str"] = new
    # file_read / file_list / image_read / str_replace_editor view: nothing else.
    return out
