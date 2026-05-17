"""RuntimeAdapter protocol — bridge UDA actions onto non-uda-desktop runtimes.

The UDA agent emits one of 17 Anthropic ``computer_use_*`` actions
(Action_20251124). On uda-desktop those flow into ``ComputerUseSandboxClient``
which talks to ``POST /v1/computer-use/action``. For runtimes that expose
a different action surface (OSWorld VMs only speak pyautogui-over-HTTP),
a ``RuntimeAdapter`` translates the same 17 actions onto that surface so
the SAME UDAAgent / TaskExecutor can drive both.

Adding a new runtime = drop a module here that implements the
:class:`RuntimeAdapter` Protocol and register it in ``__init__.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple, runtime_checkable


# ---------------------------------------------------------------------------
# xdotool keysym -> pyautogui keyname.
#
# Anthropic Computer Tool uses xdotool keysyms for ``computer_use_key`` and
# ``computer_use_hold_key``. pyautogui (which OSWorld VMs run) uses its own
# keynames (see KEYBOARD_KEYS in nanorollout/envs/desktop_env/actions.py).
# Only entries that differ from a simple ``.lower()`` are listed; the
# translator falls back to lowercase for everything else.
# ---------------------------------------------------------------------------
XDOTOOL_TO_PYAUTOGUI: Dict[str, str] = {
    "Return": "enter",
    "Escape": "esc",
    "BackSpace": "backspace",
    "Page_Up": "pageup",
    "Page_Down": "pagedown",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "Home": "home",
    "End": "end",
    "Insert": "insert",
    "Delete": "delete",
    "Tab": "tab",
    "space": "space",
    "Super_L": "winleft",
    "Super_R": "winright",
    "super": "win",
    "Control_L": "ctrlleft",
    "Control_R": "ctrlright",
    "Alt_L": "altleft",
    "Alt_R": "altright",
    "Shift_L": "shiftleft",
    "Shift_R": "shiftright",
    "Caps_Lock": "capslock",
    "Num_Lock": "numlock",
    "Scroll_Lock": "scrolllock",
    "Print": "printscreen",
}


def map_key(xkey: str) -> str:
    """Translate a single xdotool keysym to a pyautogui keyname.

    Unknown keysyms fall through as lowercase, which covers letters,
    digits, ``f1``-``f24``, ``ctrl``, ``alt``, ``shift``, etc. — the
    overwhelming majority of cases.
    """
    if xkey in XDOTOOL_TO_PYAUTOGUI:
        return XDOTOOL_TO_PYAUTOGUI[xkey]
    return xkey.lower()


def map_key_combo(xcombo: str) -> list[str]:
    """Translate an xdotool ``"ctrl+shift+c"`` combo into a pyautogui keylist."""
    return [map_key(part.strip()) for part in xcombo.split("+") if part.strip()]


# ---------------------------------------------------------------------------
# CoordScaler — maps agent-emitted pixel coords onto the target screen.
#
# UDA's tool definitions declare a viewport size (default 1920x1080 — see
# nanorollout/envs/uda_env/tools.py:36). The OSWorld VM AMI is launched
# at 1920x1080 by default (run.py:158). When the two match, scale_x/y is
# a no-op. When a model is fine-tuned on a different resolution (e.g.
# Qwen3-VL on 1280x720), the adapter is given that as ``agent_view_size``
# and rescales coords on every translation.
# ---------------------------------------------------------------------------
class CoordScaler:
    """Linear scale agent-space pixel coords to runtime-space pixel coords."""

    def __init__(
        self,
        agent_view_size: Tuple[int, int],
        screen_size: Tuple[int, int],
    ) -> None:
        self.agent_view_size = tuple(agent_view_size)
        self.screen_size = tuple(screen_size)
        aw, ah = self.agent_view_size
        sw, sh = self.screen_size
        if aw <= 0 or ah <= 0 or sw <= 0 or sh <= 0:
            raise ValueError(
                f"Invalid sizes: agent_view={self.agent_view_size}, screen={self.screen_size}"
            )
        self._sx = sw / aw
        self._sy = sh / ah
        self.is_identity = (aw == sw and ah == sh)

    def scale(self, x: float, y: float) -> Tuple[int, int]:
        """Return ``(x', y')`` rounded to the nearest pixel in screen space."""
        return int(round(x * self._sx)), int(round(y * self._sy))


# ---------------------------------------------------------------------------
# RuntimeAdapter Protocol.
#
# A RuntimeAdapter walks like a SandboxClient (TaskExecutor never calls
# ``isinstance`` on it) but does NOT inherit because its parent's __init__
# tries to build a docker/modal runtime provider. Adapters that target
# other runtimes (OSWorld AWS VM, …) implement the interface directly.
# ---------------------------------------------------------------------------
@runtime_checkable
class RuntimeAdapter(Protocol):
    """Per-runtime translator for UDA's 17 ``computer_use_*`` actions.

    Required attributes (read by the TaskExecutor for logging / metadata):

    - ``runtime_type`` (str): e.g. ``"osworld-v1"``
    - ``runtime_id`` (Optional[str]): provider-specific id, e.g. EC2 instance id
    - ``container_id`` (Optional[str]): kept None for non-docker runtimes
    - ``runtime`` (Any): driver-facing handle exposed to ``driver.score``
    - ``runtime_metadata`` (Dict): serialized into the result payload
    """

    runtime_type: str

    def create_environment(self, task: Dict[str, Any], wait_time: int) -> bool:
        """Bring the runtime up + apply task-level setup. Return True on success."""
        ...

    def cleanup_environment(self) -> bool:
        """Tear the runtime down."""
        ...

    def get_feedback(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one ``computer_use_*`` action.

        Returns the standard feedback dict shape used by ComputerUseSandboxClient::

            {"done": bool, "message": str, "image_base64"?: str}
        """
        ...

    def take_screenshot(self) -> Tuple[str, str]:
        """Capture a screenshot. Returns ``(base64_png, status_message)``."""
        ...

    def get_history(self) -> list[Dict[str, Any]]:
        """Return the per-action execution history."""
        ...

    def clear_history(self) -> None:
        """Reset the per-action execution history (called between rollouts)."""
        ...

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return a copy of runtime metadata for result serialization."""
        ...

    def health_check(self) -> bool:
        """True if the runtime is reachable."""
        ...
