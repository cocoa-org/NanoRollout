"""OSWorld v1 runtime adapter.

Bridges UDA's 17 ``computer_use_*`` actions onto the OSWorld VM's
pyautogui-over-HTTP surface (``POST {vm_ip}:5000/execute`` driven by
:class:`nanorollout.envs.desktop_env.controllers.python.PythonController`).

Lifecycle::

    create_environment(task, wait_time)
        -> DesktopEnv(provider=aws, screen=1920x1080, ...)
        -> env.reset(task_config=task["_osworld_config"])
            (runs OSWorld's setup primitives: chrome_open_tabs, launch, etc.)

    get_feedback(action)
        -> translate(action) -> pyautogui code OR short-circuit
        -> env.step(code)  [or skip for screenshot/cursor_position/zoom]

    runtime.evaluate()         # exposed for OSWorldV1Driver.score
        -> env.evaluate() -> float in [0, 1]

    cleanup_environment()
        -> env.close()        # terminates the EC2 instance

Coordinate handling: the adapter holds two sizes:

* ``agent_view_size`` — what the agent emits coords in. Defaults to
  1920x1080 to match uda-desktop's tools.py viewport declaration.
* ``screen_size`` — the OSWorld VM's actual screen. Defaults to 1920x1080
  per DesktopEnv's environment default.

When the two match (the default), :class:`CoordScaler.scale` is a no-op.
Override via ``sandbox_config["agent_view_size"]`` / ``["screen_size"]``.

Scroll units (documented so future-me doesn't second-guess):

* Anthropic ``scroll_amount`` (Action_20251124): "Number of 'clicks' to
  scroll. The clicks unit corresponds to one notch of the scroll wheel."
* pyautogui ``scroll(clicks)`` / ``hscroll(clicks)``: "clicks - The
  amount of scrolling to perform... Number of 'clicks' or scroll units."

Both are in mouse-wheel clicks. **No unit conversion**, only sign-flip:
``"up" -> +amount, "down" -> -amount, "left" -> -amount (hscroll),
"right" -> +amount (hscroll)``. ``pyautogui.hscroll`` only works on
Linux, which is fine since OSWorld VMs are Ubuntu.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..logger import get_logger
from .base import CoordScaler, map_key, map_key_combo

logger = get_logger("uda.runtime_adapter.osworld_v1")


COMPUTER_USE_PREFIX = "computer_use_"


class OSWorldV1Adapter:
    """RuntimeAdapter for OSWorld v1 (369-task suite, EC2-backed AMI).

    Implements the :class:`RuntimeAdapter` Protocol; quacks like a
    :class:`SandboxClient` so TaskExecutor can swap it in via
    ``sandbox.client_type = "osworld-v1"``.
    """

    runtime_type = "osworld-v1"

    def __init__(self, sandbox_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        cfg = dict(sandbox_config or {})
        self.sandbox_config: Dict[str, Any] = cfg

        # Resolution: agent's view vs the VM's screen. Same default by design.
        self.agent_view_size: Tuple[int, int] = tuple(
            cfg.get("agent_view_size", (1920, 1080))
        )
        self.screen_size: Tuple[int, int] = tuple(
            cfg.get("screen_size", (1920, 1080))
        )
        self.scaler = CoordScaler(self.agent_view_size, self.screen_size)

        # OSWorld DesktopEnv parameters (forwarded at create_environment time).
        self.provider_name: str = cfg.get("osworld_provider", cfg.get("provider", "aws"))
        self.region: str = cfg.get("osworld_region", "us-east-1")
        self.os_type: str = cfg.get("osworld_os_type", "Ubuntu")
        self.headless: bool = bool(cfg.get("osworld_headless", True))
        self.client_password: str = cfg.get("client_password", "")
        self.require_a11y_tree: bool = bool(cfg.get("require_a11y_tree", False))
        self.require_terminal: bool = bool(cfg.get("require_terminal", False))

        # Pacing knobs for env.step / env.evaluate.
        self.step_pause: float = float(cfg.get("step_pause", 2.0))
        self.wait_after_reset: float = float(cfg.get("wait_after_reset", 5.0))
        self.wait_before_eval: float = float(cfg.get("wait_before_eval", 5.0))

        # Standard SandboxClient surface (duck-typed; TaskExecutor reads these).
        self.container_id: Optional[str] = None
        self.runtime_id: Optional[str] = None
        self.task_name: Optional[str] = None
        self.task_dir: Optional[str] = None
        self.base_url: str = ""
        self.port: int = 0
        self.llm_provider: Optional[str] = (
            cfg.get("llm_provider")
            or os.getenv("UDA_LLM_PROVIDER")
            or os.getenv("COCOA_LLM_PROVIDER")
        )
        self.llm_model: Optional[str] = (
            cfg.get("llm_model")
            or os.getenv("UDA_LLM_MODEL")
            or os.getenv("COCOA_LLM_MODEL")
        )
        self.runtime_metadata: Dict[str, Any] = {
            "type": self.runtime_type,
            "provider": self.provider_name,
            "region": self.region,
            "surfaces": ["computer-use", "evaluate"],
            "agent_view_size": list(self.agent_view_size),
            "screen_size": list(self.screen_size),
        }

        # Set on create_environment.
        self.env = None  # type: ignore[assignment]
        # ``runtime`` is the handle BenchDriver.score(runtime, ...) receives.
        # We hand the driver the env itself so it can call env.evaluate().
        self.runtime = None  # type: ignore[assignment]

        # Per-action execution history (action + feedback summary).
        self.execution_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # SandboxClient-compatible lifecycle.
    # ------------------------------------------------------------------ #

    def create_environment(self, task: Dict[str, Any], wait_time: int = 60) -> bool:
        """Launch the OSWorld VM and run its setup primitives.

        The task must carry ``_osworld_config`` — the verbatim OSWorld task
        JSON (id / instruction / config / evaluator). The UDA bench runner
        attaches this when ``bench == "osworld-v1"``.
        """
        osworld_config = task.get("_osworld_config")
        if not isinstance(osworld_config, dict):
            raise ValueError(
                "OSWorldV1Adapter: task is missing '_osworld_config'. "
                "Ensure the uda runner attaches the OSWorld task JSON for "
                "bench='osworld-v1'."
            )

        # Lazy import: pulls boto3 + DesktopEnv, ~seconds of cold start.
        from nanorollout.envs.desktop_env.desktop_env import DesktopEnv

        logger.info(
            "OSWorldV1Adapter: launching %s VM (region=%s, screen=%dx%d)",
            self.provider_name,
            self.region,
            self.screen_size[0],
            self.screen_size[1],
        )
        self.env = DesktopEnv(
            provider_name=self.provider_name,
            region=self.region,
            os_type=self.os_type,
            action_space="pyautogui",
            headless=self.headless,
            require_a11y_tree=self.require_a11y_tree,
            require_terminal=self.require_terminal,
            screen_size=self.screen_size,
            client_password=self.client_password,
        )
        self.runtime = self.env
        self.runtime_id = getattr(self.env, "path_to_vm", None)
        self.runtime_metadata["runtime_id"] = self.runtime_id
        self.runtime_metadata["vm_ip"] = getattr(self.env, "vm_ip", None)

        logger.info(
            "OSWorldV1Adapter: env.reset(id=%s)",
            osworld_config.get("id", "?"),
        )
        self.env.reset(task_config=osworld_config)
        if self.wait_after_reset > 0:
            time.sleep(self.wait_after_reset)
        self.task_name = osworld_config.get("id")
        self.task_dir = None
        logger.info("OSWorldV1Adapter: ready (instance=%s)", self.runtime_id)
        return True

    def cleanup_environment(self) -> bool:
        """Tear the EC2 instance down."""
        if self.env is None:
            return True
        try:
            self.env.close()
            logger.info("OSWorldV1Adapter: env closed (instance=%s)", self.runtime_id)
        except Exception as exc:
            logger.warning("OSWorldV1Adapter: env.close() raised: %s", exc)
            return False
        finally:
            self.env = None
            self.runtime = None
        return True

    def get_runtime_metadata(self) -> Dict[str, Any]:
        return dict(self.runtime_metadata)

    def health_check(self) -> bool:
        if self.env is None:
            return False
        # The cheapest live-VM check is /screen_size; a successful response
        # also confirms the in-VM pyautogui server is up.
        try:
            size = self.env.controller.get_vm_screen_size()
            return bool(size)
        except Exception:
            return False

    def get_history(self) -> List[Dict[str, Any]]:
        return self.execution_history

    def clear_history(self) -> None:
        self.execution_history = []

    # ------------------------------------------------------------------ #
    # Action dispatch.
    # ------------------------------------------------------------------ #

    def get_feedback(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a single UDA action onto the OSWorld VM.

        Routing:

        * ``computer_use_screenshot / cursor_position / zoom`` — short-circuit;
          adapter handles them via /screenshot or /run_python rather than
          going through env.step.
        * ``computer_use_wait`` — env.step("WAIT", pause=duration).
        * Other ``computer_use_*`` — translated to a single pyautogui code
          string, dispatched via env.step.
        * ``task_complete`` / ``exit`` — env.step("DONE"), feedback.done=True.
        """
        action_type = action.get("action_type", "")
        feedback: Dict[str, Any]
        try:
            if action_type in ("task_complete", "exit"):
                feedback = self._handle_done(action)
            elif action_type == "computer_use_screenshot":
                feedback = self._handle_screenshot()
            elif action_type == "computer_use_cursor_position":
                feedback = self._handle_cursor_position()
            elif action_type == "computer_use_zoom":
                feedback = self._handle_zoom(action)
            elif action_type == "computer_use_wait":
                feedback = self._handle_wait(action)
            elif isinstance(action_type, str) and action_type.startswith(COMPUTER_USE_PREFIX):
                feedback = self._handle_pyautogui(action)
            else:
                feedback = {
                    "done": False,
                    "message": (
                        f"OSWorldV1Adapter: unsupported action_type={action_type!r}. "
                        "Only computer_use_* + task_complete are honored on the OSWorld runtime."
                    ),
                }
        except Exception as exc:
            logger.exception("OSWorldV1Adapter: action %s failed", action_type)
            feedback = {"done": False, "message": f"OSWorldV1Adapter error: {exc}"}

        self.execution_history.append({"action": action, "feedback": _summarize(feedback)})
        return feedback

    def take_screenshot(self) -> Tuple[str, str]:
        """Return ``(base64_png, status_message)`` for a fresh screenshot."""
        b64 = self._screenshot_b64()
        if b64:
            return b64, f"Screenshot taken successfully ({len(b64)} chars base64)"
        return "", "Failed to take screenshot"

    # ------------------------------------------------------------------ #
    # Action handlers — one per UDA action_type.
    # ------------------------------------------------------------------ #

    def _handle_done(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """``task_complete`` / ``exit`` → env.step("DONE"); end the rollout."""
        try:
            self.env.step("DONE", pause=0.5)
        except Exception:
            logger.exception("OSWorldV1Adapter: env.step('DONE') failed")
        return {"done": True, "message": "OSWorldV1Adapter: task marked complete"}

    def _handle_screenshot(self) -> Dict[str, Any]:
        b64 = self._screenshot_b64()
        if not b64:
            return {"done": False, "message": "computer_use_screenshot: failed"}
        return {
            "done": False,
            "message": f"computer_use_screenshot: ok ({len(b64)} chars base64 image)",
            "image_base64": b64,
        }

    def _handle_cursor_position(self) -> Dict[str, Any]:
        """``computer_use_cursor_position`` — run pyautogui.position() in VM."""
        script = "import pyautogui;p=pyautogui.position();print(int(p.x),int(p.y))"
        result = self.env.controller.run_python_script(script) or {}
        output = (result.get("output") or "").strip()
        m = re.search(r"(-?\d+)\s+(-?\d+)", output)
        if not m:
            return {
                "done": False,
                "message": (
                    f"computer_use_cursor_position: failed to parse "
                    f"(stdout={output!r}, error={result.get('error')!r})"
                ),
            }
        x, y = int(m.group(1)), int(m.group(2))
        return {
            "done": False,
            "message": f"computer_use_cursor_position: ({x}, {y})",
        }

    def _handle_zoom(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """``computer_use_zoom`` — crop region from a fresh screenshot."""
        region = action.get("region")
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            return {
                "done": False,
                "message": "computer_use_zoom: region must be a 4-tuple [x1,y1,x2,y2]",
            }
        x1, y1, x2, y2 = (int(v) for v in region)
        # Region is emitted in agent-view space; scale into screen space.
        x1, y1 = self.scaler.scale(x1, y1)
        x2, y2 = self.scaler.scale(x2, y2)
        if x2 <= x1 or y2 <= y1:
            return {
                "done": False,
                "message": f"computer_use_zoom: empty/inverted region after scaling: ({x1},{y1},{x2},{y2})",
            }

        raw = self.env.controller.get_screenshot()
        if not raw:
            return {"done": False, "message": "computer_use_zoom: failed to capture screen"}

        try:
            img = Image.open(io.BytesIO(raw))
            cropped = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            return {"done": False, "message": f"computer_use_zoom: crop failed: {exc}"}

        return {
            "done": False,
            "message": f"computer_use_zoom: ok ({x2 - x1}x{y2 - y1} crop, {len(b64)} chars base64)",
            "image_base64": b64,
        }

    def _handle_wait(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """``computer_use_wait`` — OSWorld has a native "WAIT" sentinel."""
        duration = float(action.get("duration") or 1.0)
        # env.step("WAIT", pause=duration) sleeps then takes the obs.
        self.env.step("WAIT", pause=duration)
        return {"done": False, "message": f"computer_use_wait: slept {duration}s"}

    def _handle_pyautogui(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Generic path: translate a UDA action into pyautogui code + env.step."""
        action_type = action["action_type"]
        anthropic_name = action_type[len(COMPUTER_USE_PREFIX):]
        code = self._translate(anthropic_name, action)
        if not code:
            return {
                "done": False,
                "message": f"OSWorldV1Adapter: no translation for {action_type}",
            }
        logger.debug("OSWorldV1Adapter: %s -> %s", action_type, code)
        self.env.step(code, pause=self.step_pause)
        return {
            "done": False,
            "message": f"{action_type}: dispatched ({code[:120]}{'...' if len(code) > 120 else ''})",
        }

    # ------------------------------------------------------------------ #
    # Translator — the 17-row table.
    # ------------------------------------------------------------------ #

    def _translate(self, anthropic_name: str, action: Dict[str, Any]) -> Optional[str]:
        """Map a single ``computer_use_*`` action to a one-line pyautogui code.

        Returns None for actions that should not reach env.step (screenshot,
        cursor_position, zoom, wait, task_complete) — those are short-circuited
        upstream by :meth:`get_feedback`.
        """

        coord = action.get("coordinate")
        start = action.get("start_coordinate")

        # ---- mouse movement / clicks ---------------------------------------
        if anthropic_name == "mouse_move":
            x, y = self._sxy(coord)
            return f"pyautogui.moveTo({x}, {y})"

        if anthropic_name == "left_click":
            mods = action.get("text")
            click_call = self._click_call("click", coord)
            if mods:
                return self._wrap_with_modifiers(mods, click_call)
            return click_call

        if anthropic_name == "right_click":
            return self._click_call("rightClick", coord)

        if anthropic_name == "middle_click":
            return self._click_call("middleClick", coord)

        if anthropic_name == "double_click":
            return self._click_call("doubleClick", coord)

        if anthropic_name == "triple_click":
            return self._click_call("tripleClick", coord)

        if anthropic_name == "left_click_drag":
            if not (isinstance(start, (list, tuple)) and isinstance(coord, (list, tuple))):
                logger.warning("left_click_drag missing start_coordinate/coordinate")
                return None
            sx, sy = self._sxy(start)
            ex, ey = self._sxy(coord)
            duration = float(action.get("duration") or 0.5)
            return (
                f"pyautogui.moveTo({sx}, {sy}); "
                f"pyautogui.dragTo({ex}, {ey}, duration={duration}, button='left')"
            )

        if anthropic_name == "left_mouse_down":
            if coord:
                x, y = self._sxy(coord)
                return f"pyautogui.moveTo({x}, {y}); pyautogui.mouseDown(button='left')"
            return "pyautogui.mouseDown(button='left')"

        if anthropic_name == "left_mouse_up":
            if coord:
                x, y = self._sxy(coord)
                return f"pyautogui.moveTo({x}, {y}); pyautogui.mouseUp(button='left')"
            return "pyautogui.mouseUp(button='left')"

        # ---- keyboard ------------------------------------------------------
        if anthropic_name == "key":
            text = action.get("text", "")
            keys = map_key_combo(text) if isinstance(text, str) else []
            if not keys:
                logger.warning("computer_use_key: empty text=%r", text)
                return None
            keys_repr = ", ".join(repr(k) for k in keys)
            # pyautogui.hotkey handles N>=1 keys (down-in-order, up-in-reverse).
            return f"pyautogui.hotkey({keys_repr})"

        if anthropic_name == "type":
            text = action.get("text", "")
            # interval=0.01 avoids dropped keys on slow X servers.
            return f"pyautogui.typewrite({text!r}, interval=0.01)"

        if anthropic_name == "hold_key":
            text = action.get("text") or action.get("key")
            duration = float(action.get("duration") or 1.0)
            if not text:
                return None
            keys = map_key_combo(text)
            if not keys:
                return None
            downs = "; ".join(f"pyautogui.keyDown({k!r})" for k in keys)
            ups = "; ".join(f"pyautogui.keyUp({k!r})" for k in reversed(keys))
            return f"{downs}; time.sleep({duration}); {ups}"

        # ---- scroll --------------------------------------------------------
        if anthropic_name == "scroll":
            direction = (action.get("scroll_direction") or "down").lower()
            amount = int(action.get("scroll_amount") or 1)
            prefix = ""
            if coord:
                x, y = self._sxy(coord)
                prefix = f"pyautogui.moveTo({x}, {y}); "
            if direction in ("up", "down"):
                signed = amount if direction == "up" else -amount
                return f"{prefix}pyautogui.scroll({signed})"
            if direction in ("left", "right"):
                signed = amount if direction == "right" else -amount
                return f"{prefix}pyautogui.hscroll({signed})"
            logger.warning("computer_use_scroll: unknown direction=%r", direction)
            return None

        logger.warning("OSWorldV1Adapter: unhandled action %r", anthropic_name)
        return None

    # ------------------------------------------------------------------ #
    # Per-runtime evaluation (called by OSWorldV1Driver.score).
    # ------------------------------------------------------------------ #

    def evaluate(self) -> float:
        """Run OSWorld's bundled evaluator (returns float in [0, 1])."""
        if self.env is None:
            raise RuntimeError("OSWorldV1Adapter.evaluate(): no live env")
        if self.wait_before_eval > 0:
            time.sleep(self.wait_before_eval)
        return float(self.env.evaluate())

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #

    def _sxy(self, coord: Any) -> Tuple[int, int]:
        """Scale and unpack a 2-tuple coordinate; (0, 0) on invalid input."""
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return 0, 0
        return self.scaler.scale(int(coord[0]), int(coord[1]))

    def _click_call(self, fn: str, coord: Any) -> str:
        """Render ``pyautogui.{fn}({x}, {y})`` or ``pyautogui.{fn}()`` if no coord."""
        if coord:
            x, y = self._sxy(coord)
            return f"pyautogui.{fn}({x}, {y})"
        return f"pyautogui.{fn}()"

    @staticmethod
    def _wrap_with_modifiers(mods_text: str, inner: str) -> str:
        """Hold ``mods_text`` modifier keys for the duration of ``inner``.

        Anthropic spec: ``text`` on left_click is a ``+``-joined modifier
        list (e.g. ``"ctrl+shift"``) held during the click. Mirror that:
        keyDown all -> inner -> keyUp all in reverse.
        """
        mods = map_key_combo(mods_text)
        if not mods:
            return inner
        downs = "; ".join(f"pyautogui.keyDown({k!r})" for k in mods)
        ups = "; ".join(f"pyautogui.keyUp({k!r})" for k in reversed(mods))
        return f"{downs}; {inner}; {ups}"

    def _screenshot_b64(self) -> str:
        if self.env is None:
            return ""
        raw = self.env.controller.get_screenshot()
        if not raw:
            return ""
        return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Drop large fields (image_base64) before recording into history."""
    out = {k: v for k, v in feedback.items() if k != "image_base64"}
    if "image_base64" in feedback:
        out["image_bytes"] = len(feedback["image_base64"])
    return out
