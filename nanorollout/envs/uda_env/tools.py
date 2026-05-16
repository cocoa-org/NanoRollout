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


def get_computer_use_tools() -> List[Dict[str, Any]]:
    """Get OpenAI tool definitions for the computer-use action set.

    Returns:
        List of 17 tools, one per Anthropic Computer Tool action, plus
        the shared ``task_complete`` sentinel.
    """
    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "computer_use_screenshot",
                "description": f"{_COMPUTER_USE_DESCRIPTION} Capture the current screen as a base64-encoded PNG.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_cursor_position",
                "description": "Return the current mouse cursor (x, y) in the X11 root window.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_mouse_move",
                "description": "Move the mouse pointer to the given (x, y) without clicking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Target pixel coordinate [x, y]; (0, 0) is the top-left of the root window.",
                        }
                    },
                    "required": ["coordinate"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_left_click",
                "description": "Left-click. Optionally move to `coordinate` first; optionally hold a modifier `key` (e.g. 'ctrl', 'shift').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional pixel coordinate to move to before clicking.",
                        },
                        "key": {
                            "type": "string",
                            "description": "Optional modifier to hold during the click (xdotool keysym, e.g. 'ctrl', 'shift', 'alt').",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_right_click",
                "description": "Right-click. Same parameters as `computer_use_left_click`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional pixel coordinate to move to before clicking.",
                        },
                        "key": {"type": "string", "description": "Optional modifier keysym."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_middle_click",
                "description": "Middle-click. Same parameters as `computer_use_left_click`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional pixel coordinate to move to before clicking.",
                        },
                        "key": {"type": "string", "description": "Optional modifier keysym."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_double_click",
                "description": "Double-click. Same parameters as `computer_use_left_click`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional pixel coordinate to move to before clicking.",
                        },
                        "key": {"type": "string", "description": "Optional modifier keysym."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_triple_click",
                "description": "Triple-click (selects a paragraph/line in most editors). Same parameters as `computer_use_left_click`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional pixel coordinate to move to before clicking.",
                        },
                        "key": {"type": "string", "description": "Optional modifier keysym."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_left_click_drag",
                "description": "Press the left mouse button at `start_coordinate`, drag to `coordinate`, then release. Use for selecting text / dragging windows / drawing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Drag start pixel coordinate [x, y].",
                        },
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Drag end pixel coordinate [x, y].",
                        },
                    },
                    "required": ["start_coordinate", "coordinate"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_left_mouse_down",
                "description": "Press and hold the left mouse button (no release). Use only when chaining manual drags.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_left_mouse_up",
                "description": "Release the left mouse button.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_key",
                "description": "Press a single key or chord. `text` is an xdotool keysym (e.g. 'Return', 'Tab', 'ctrl+c', 'super+l').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "xdotool keysym, e.g. 'Return', 'Escape', 'ctrl+c'.",
                        }
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_type",
                "description": "Type the given UTF-8 string into the currently focused window (xdotool type).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to type."}
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_hold_key",
                "description": "Press a key, sleep `duration` seconds, then release. Useful for sticky modifiers or game-style holds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "xdotool keysym to hold."},
                        "duration": {
                            "type": "number",
                            "description": "Seconds to hold the key (0–100).",
                        },
                    },
                    "required": ["text", "duration"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_scroll",
                "description": "Scroll in `scroll_direction` by `scroll_amount` ticks. Optionally move to `coordinate` first; optionally hold modifier `text` during the scroll.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scroll_direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Scroll direction.",
                        },
                        "scroll_amount": {
                            "type": "integer",
                            "description": "Number of scroll-wheel ticks (>= 0).",
                        },
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional pixel coordinate to move to before scrolling.",
                        },
                        "text": {
                            "type": "string",
                            "description": "Optional modifier keysym to hold during the scroll (e.g. 'ctrl' for zoom).",
                        },
                    },
                    "required": ["scroll_direction", "scroll_amount"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_wait",
                "description": "Sleep `duration` seconds, then take a screenshot. Use for UI animations / dialogs settling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration": {
                            "type": "number",
                            "description": "Seconds to wait before taking the post-screenshot.",
                        }
                    },
                    "required": ["duration"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "computer_use_zoom",
                "description": "Take a screenshot of a sub-region of the screen (returns the crop in base64_image).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Crop rect [x0, y0, x1, y1]; must have x1>x0 and y1>y0.",
                        }
                    },
                    "required": ["region"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": "Mark the task as complete and exit. Optionally provide the final result/answer if the task requires returning a specific output (e.g., JSON answer). For tasks that generate files in the sandbox, you can omit the result parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "Optional: The final result or answer for the task (e.g., JSON string). Use this when the task requires returning a specific output. For tasks that generate files, omit this parameter.",
                        }
                    },
                },
            },
        },
    ]
    return tools


def get_file_tools() -> List[Dict[str, Any]]:
    """Get OpenAI tool definitions for file operations.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_list",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the directory to list"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": "Replace text in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Absolute path to the file"
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Text to replace (will be converted to old_str for API)"
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text (will be converted to new_str for API)"
                        }
                    },
                    "required": ["file", "old_text", "new_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_in_file",
                "description": "Search for text in a file using regex pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Absolute path to the file"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Regular expression pattern to search for (will be converted to regex for API)"
                        }
                    },
                    "required": ["file", "pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_files",
                "description": "Find files matching a glob pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to search in"
                        },
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern (e.g., '*.py', '**/*.txt')"
                        }
                    },
                    "required": ["path", "glob"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "image_read",
                "description": "Read an image file (PNG, JPG, etc.) and return it as base64-encoded image for visual analysis. Use this to read visualization files generated by code (e.g., matplotlib plots, saved figures). The image will be automatically included in subsequent prompts for analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the image file to read (supports PNG, JPG, JPEG formats)"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Advanced file editor with view, create, str_replace, insert, undo_edit commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                            "description": "Editor command to execute"
                        },
                        "path": {
                            "type": "string",
                            "description": "Absolute path to file or directory"
                        },
                        "file_text": {
                            "type": "string",
                            "description": "File content for 'create' command"
                        },
                        "old_str": {
                            "type": "string",
                            "description": "String to replace for 'str_replace' command"
                        },
                        "new_str": {
                            "type": "string",
                            "description": "New string for 'str_replace' or 'insert' command"
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": "Line number for 'insert' command"
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Line range for 'view' command [start, end]"
                        }
                    },
                    "required": ["command", "path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": "Mark the task as complete and exit. Optionally provide the final result/answer if the task requires returning a specific output (e.g., JSON answer). For tasks that generate files in the sandbox, you can omit the result parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "Optional: The final result or answer for the task (e.g., JSON string). Use this when the task requires returning a specific output. For tasks that generate files, omit this parameter."
                        }
                    }
                }
            }
        }
    ]

    return tools


def get_code_tools() -> List[Dict[str, Any]]:
    """Get OpenAI tool definitions for code execution.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "code_execute",
                "description": "Execute code via sandbox runtime (python default). Returns stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Source code to execute"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript"],
                            "description": "Runtime language (default python)"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Optional timeout in seconds"
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": "Mark the task as complete and exit. Optionally provide the final result/answer if the task requires returning a specific output (e.g., JSON answer). For tasks that generate files in the sandbox, you can omit the result parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "Optional: The final result or answer for the task (e.g., JSON string). Use this when the task requires returning a specific output. For tasks that generate files, omit this parameter."
                        }
                    }
                }
            }
        }
    ]

    return tools


def get_shell_tools() -> List[Dict[str, Any]]:
    """Get OpenAI tool definitions for shell operations.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell_execute",
                "description": "Execute a shell command and get the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": "Mark the task as complete and exit. Optionally provide the final result/answer if the task requires returning a specific output (e.g., JSON answer). For tasks that generate files in the sandbox, you can omit the result parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "Optional: The final result or answer for the task (e.g., JSON string). Use this when the task requires returning a specific output. For tasks that generate files, omit this parameter."
                        }
                    }
                }
            }
        }
    ]

    return tools


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


# Valid parameter sets per tool (catches param typos / unsupported args early).
# Computer-use entries mirror server.py's ActionRequest pydantic schema.
_COMPUTER_USE_VALID_PARAMS: Dict[str, set] = {
    "computer_use_screenshot": set(),
    "computer_use_cursor_position": set(),
    "computer_use_mouse_move": {"coordinate"},
    "computer_use_left_click": {"coordinate", "key"},
    "computer_use_right_click": {"coordinate", "key"},
    "computer_use_middle_click": {"coordinate", "key"},
    "computer_use_double_click": {"coordinate", "key"},
    "computer_use_triple_click": {"coordinate", "key"},
    "computer_use_left_click_drag": {"start_coordinate", "coordinate"},
    "computer_use_left_mouse_down": set(),
    "computer_use_left_mouse_up": set(),
    "computer_use_key": {"text"},
    "computer_use_type": {"text"},
    "computer_use_hold_key": {"text", "duration"},
    "computer_use_scroll": {"scroll_direction", "scroll_amount", "coordinate", "text"},
    "computer_use_wait": {"duration"},
    "computer_use_zoom": {"region"},
}


def map_tool_call_to_action(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Map an LLM tool call to a sandbox action.

    The action's ``action_type`` is the tool name verbatim — dispatch in
    :class:`UnifiedSandboxClient` recognises ``computer_use_*`` /
    ``file_*`` / ``code_*`` / ``shell_*`` prefixes and routes accordingly.

    Args:
        tool_name: Name of the tool being called (e.g. ``computer_use_left_click``).
        arguments: Tool arguments as emitted by the LLM.

    Returns:
        Action dictionary for the sandbox client.

    Raises:
        ValueError: If ``tool_name`` is unknown or ``arguments`` contains
            keys not in the tool's whitelisted parameter set.
    """
    tool_valid_params: Dict[str, set] = {
        **_COMPUTER_USE_VALID_PARAMS,
        "file_read": {"path"},
        "file_write": {"path", "content"},
        "file_list": {"path"},
        "replace_in_file": {"file", "old_text", "new_text"},
        "search_in_file": {"file", "pattern"},
        "find_files": {"path", "glob"},
        "image_read": {"path"},
        "str_replace_editor": {"command", "path", "file_text", "old_str", "new_str", "insert_line", "view_range"},
        "code_execute": {"code", "language", "timeout"},
        "shell_execute": {"command"},
        "task_complete": {"result"},
    }

    if tool_name not in tool_valid_params:
        raise ValueError(f"Unknown tool: {tool_name}")

    valid_params = tool_valid_params[tool_name]
    invalid_params = set(arguments.keys()) - valid_params
    if invalid_params:
        raise ValueError(
            f"Tool '{tool_name}' does not support parameters: {invalid_params}. "
            f"Valid parameters are: {valid_params}. "
            f"Received: {list(arguments.keys())}"
        )
    arguments = {k: v for k, v in arguments.items() if k in valid_params}

    action: Dict[str, Any] = {"action_type": tool_name}
    action.update(arguments)
    return action
