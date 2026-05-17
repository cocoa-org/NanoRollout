"""UDA sandbox HTTP clients and shared runtime primitives.

This is a UDA-specific reimplementation of cocoa_env.base: the browser
layer (cocoa's CDP-driven DOM-aware ``BrowserSandboxClient``, ~1500
lines) is **replaced** by :class:`ComputerUseSandboxClient`, a pixel
-level client targeting uda-desktop's ``/v1/computer-use/*`` HTTP
surface (Anthropic Computer Tool, 17 actions). File / code / shell /
jupyter clients still ride the agent-infra/sandbox SDK because
uda-desktop inherits those endpoints unchanged on ``:8080``.

LLM-provider env vars: UDA reads ``UDA_LLM_PROVIDER`` / ``UDA_LLM_MODEL``
first and falls back to the legacy ``COCOA_LLM_*`` names for users with
existing configs.

Default workspace is ``/home/kasm-user`` (the UID-1000 user inside the
uda-desktop image). A ``/home/gem -> /home/kasm-user`` symlink in the
image catches any path strings missed by migration.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from PIL import Image
from agent_sandbox import Sandbox

from .utils import retry_request, validate_response, get_logger, colorize

runtime_logger = get_logger("uda.runtime")
logger = get_logger("uda.sandbox")

UDA_WORKSPACE = "/home/kasm-user"
COMPUTER_USE_ACTION_PREFIX = "computer_use_"


class BaseSandboxRuntime:
    """Lifecycle manager for a sandbox runtime backend.

    Concrete backends (docker, modal) need only implement ``start`` and
    ``cleanup``. The cross-runtime data primitives — pushing files into
    the container and running shell commands — have a default impl here
    that goes through the agent-infra/sandbox SDK's HTTP surface
    (``/v1/file``, ``/v1/shell``). That surface is exposed identically
    by both runtimes (docker forwards ``:8080`` to localhost; modal
    tunnels it). Per-runtime subclasses are free to override with a
    faster native path (docker uses ``docker cp``).
    """

    runtime_type = "base"

    def __init__(self, client: Any):
        self.client = client

    def start(self, task: Dict[str, Any], wait_time: int = 60) -> bool:
        """Start a sandbox for the provided task."""
        raise NotImplementedError

    def cleanup(self) -> bool:
        """Stop and clean up the current sandbox."""
        raise NotImplementedError

    def copy_to_runtime(self, host_path: str, container_path: str) -> bool:
        """Copy a file or directory from host into the container.

        Default impl: walk ``host_path`` and ``sdk_client.file.write_file``
        each file in turn. Directories are created implicitly. Returns
        False on any failure. Override for faster native transports.
        """
        host = Path(host_path)
        if not host.exists():
            runtime_logger.error("copy_to_runtime: source missing: %s", host)
            return False
        sdk = self._ensure_sdk_client()
        if sdk is None:
            runtime_logger.error(
                "copy_to_runtime: could not initialize sdk_client on %s",
                type(self.client).__name__,
            )
            return False
        try:
            if host.is_file():
                self._write_single_file(sdk, host, container_path)
            else:
                # Mirror the directory tree at container_path.
                base = host.resolve()
                for src in host.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.resolve().relative_to(base).as_posix()
                    dest = f"{container_path.rstrip('/')}/{rel}"
                    self._write_single_file(sdk, src, dest)
            return True
        except Exception as exc:
            runtime_logger.error(
                "copy_to_runtime failed (%s -> %s): %s", host, container_path, exc
            )
            return False

    def _ensure_sdk_client(self):
        """Lazily initialise the agent-infra/sandbox SDK client.

        Driver hooks (``setup_workspace``, ``run_warmup``) run BEFORE the
        agent rollout starts, which is when ``get_feedback`` would
        otherwise trigger ``_initialize_sdk_client``. Without this lazy
        init, the SDK-mediated default ``copy_to_runtime`` /
        ``exec_in_runtime`` calls would crash on modal (docker has its
        own native ``docker cp`` override and doesn't hit this path).
        """
        client = self.client
        init_fn = getattr(client, "_initialize_sdk_client", None)
        if init_fn is not None and getattr(client, "sdk_client", None) is None:
            try:
                init_fn()
            except Exception as exc:
                runtime_logger.error("sdk_client lazy init failed: %s", exc)
                return None
        return getattr(client, "sdk_client", None)

    @staticmethod
    def _write_single_file(sdk, src: "Path", dest: str) -> None:
        """SDK-mediated single-file write into the container."""
        with open(src, "rb") as fh:
            data = fh.read()
        # The agent-infra sandbox accepts text or base64-encoded bytes
        # depending on SDK version; use bytes when available, else base64.
        try:
            sdk.file.write_file(file=dest, content=data)
        except TypeError:
            sdk.file.write_file(file=dest, content=base64.b64encode(data).decode("ascii"))

    def exec_in_runtime(
        self,
        command: str,
        *,
        workdir: Optional[str] = None,
        timeout: int = 600,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run a shell command inside the container.

        Returns ``{"output": str, "returncode": int}``. The returncode
        is best-effort — the SDK only surfaces stdout text for some
        backends, so non-zero is inferred from the absence of output +
        any thrown SDK exception.

        ``env`` is inline-prefixed bash-style (``K1='v1' K2='v2' cmd``)
        so the command sees those env vars regardless of whether the
        underlying shell session inherits them. Values are shell-quoted.

        Default impl uses ``sdk_client.shell.exec_command`` so it works
        identically on docker (port-forwarded :8080) and modal (tunnel).
        """
        sdk = self._ensure_sdk_client()
        if sdk is None:
            return {
                "output": "",
                "returncode": -1,
                "error": "could not initialize sdk_client on runtime client",
            }
        if env:
            import shlex as _shlex
            prefix = " ".join(f"{k}={_shlex.quote(str(v))}" for k, v in env.items())
            command = f"{prefix} {command}"
        try:
            session = sdk.shell.create_session(exec_dir=workdir or UDA_WORKSPACE)
            session_id = session.data.session_id
            result = sdk.shell.exec_command(
                command=command,
                id=session_id,
                exec_dir=workdir or UDA_WORKSPACE,
                async_mode=False,
                timeout=int(timeout),
            )
            output = getattr(result.data, "output", "") or ""
            return {"output": output, "returncode": 0}
        except Exception as exc:
            runtime_logger.error("exec_in_runtime failed: %s", exc)
            return {"output": "", "returncode": -1, "error": str(exc)}

    def metadata(self) -> Dict[str, Any]:
        """Return provider-specific runtime metadata."""
        return {"type": self.runtime_type}

    def _wait_for_health(self, wait_time: int) -> bool:
        waited = 0
        sleep_interval = 5
        while waited < wait_time:
            if self.client.health_check():
                return True
            waited += sleep_interval
            runtime_logger.info(
                "Sandbox not ready yet. Waiting ... (%s/%s seconds)",
                waited,
                wait_time,
            )
            time.sleep(sleep_interval)
        return False


class SandboxClient:
    """Base HTTP client for the UDA sandbox.

    Talks to a single ``:8080`` origin that aggregates
    agent-infra/sandbox's ``/v1/shell``, ``/v1/file``, ``/v1/code``,
    ``/v1/jupyter``, ``/v1/browser`` (deprecated for UDA), and
    uda-desktop's ``/v1/computer-use`` endpoints.
    """

    def __init__(self, sandbox_config: Dict[str, Any] | None = None, **kwargs):
        if sandbox_config is None:
            sandbox_config = {}
        self.sandbox_config = dict(sandbox_config)

        self.port = self.sandbox_config.get("docker_port", kwargs.get("port", 8080))
        base_url = self.sandbox_config.get(
            "base_url", kwargs.get("base_url", f"http://localhost:{self.port}")
        )

        self.base_url = base_url.rstrip("/")
        self.container_id: Optional[str] = None
        self.runtime_id: Optional[str] = None
        self.task_name: Optional[str] = None
        self.task_dir: Optional[str] = None

        # LLM provider/model identification — prefer UDA_* env vars,
        # fall back to legacy COCOA_* for backward-compat.
        self.llm_provider = (
            self.sandbox_config.get("llm_provider")
            or os.getenv("UDA_LLM_PROVIDER")
            or os.getenv("COCOA_LLM_PROVIDER")
        )
        self.llm_model = (
            self.sandbox_config.get("llm_model")
            or os.getenv("UDA_LLM_MODEL")
            or os.getenv("COCOA_LLM_MODEL")
        )

        self.runtime_type = (self.sandbox_config.get("runtime_type") or "docker").strip().lower()
        self.runtime_metadata: Dict[str, Any] = {
            "type": self.runtime_type,
            "base_url": self.base_url,
            "surfaces": ["shell", "file", "code", "jupyter", "computer-use"],
        }
        # Optional UDA-specific identification stamped by the runner.
        uda_image = self.sandbox_config.get("uda_image")
        if uda_image:
            self.runtime_metadata["uda_image"] = uda_image
        bench = self.sandbox_config.get("bench")
        if bench:
            self.runtime_metadata["bench"] = bench
        corpus_revision = self.sandbox_config.get("corpus_revision")
        if corpus_revision:
            self.runtime_metadata["corpus_revision"] = corpus_revision

        self.runtime = self._create_runtime_provider()

    def _create_runtime_provider(self):
        """Construct the configured runtime backend."""
        from .docker import DockerComposeSandboxRuntime
        from .modal import ModalSandboxRuntime

        providers = {
            "docker": DockerComposeSandboxRuntime,
            "modal": ModalSandboxRuntime,
        }
        provider_cls = providers.get(self.runtime_type)
        if provider_cls is None:
            raise ValueError(
                f"Unsupported sandbox runtime_type='{self.runtime_type}'. "
                f"Supported values: {sorted(providers)}"
            )
        return provider_cls(self)

    def _update_runtime_metadata(self, **kwargs: Any) -> None:
        """Update runtime metadata with non-null values."""
        for key, value in kwargs.items():
            if value is not None:
                self.runtime_metadata[key] = value

    def set_base_url(self, base_url: str) -> None:
        """Update the active sandbox base URL."""
        self.base_url = base_url.rstrip("/")
        self._update_runtime_metadata(base_url=self.base_url)

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return a copy of runtime metadata for result serialization."""
        return dict(self.runtime_metadata)

    def _should_compress_for_claude(self) -> bool:
        provider = (getattr(self, "llm_provider", None) or "").lower()
        model = (getattr(self, "llm_model", None) or "").lower()
        return provider == "claude" or "claude" in model

    def _is_kimi_model(self) -> bool:
        """Return True when the current sandbox is serving a Kimi controller."""
        provider = (getattr(self, "llm_provider", None) or "").strip().lower()
        model = (getattr(self, "llm_model", None) or "").strip().lower()
        return provider == "kimi" or "kimi" in model or "moonshot" in model

    def _is_qwen3_model(self) -> bool:
        """Return True when the current sandbox is serving a Qwen3-family controller."""
        provider = (getattr(self, "llm_provider", None) or "").strip().lower()
        model = (getattr(self, "llm_model", None) or "").strip().lower()
        return "qwen3" in model or (provider == "qwen" and "qwen" in model)

    def _compress_image_bytes_for_claude(
        self, image_bytes: bytes, max_base64_bytes: int = 5 * 1024 * 1024
    ) -> bytes:
        if not self._should_compress_for_claude():
            return image_bytes
        if len(base64.b64encode(image_bytes)) <= max_base64_bytes:
            return image_bytes
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Cannot open image for compression and it exceeds size limit: {e}") from e
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        width, height = img.size
        max_dim = 1568
        scale = min(1.0, max_dim / max(width, height)) if max(width, height) > max_dim else 1.0
        quality = 85
        data = image_bytes
        for _ in range(32):
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            resized = img.resize((new_w, new_h)) if (new_w, new_h) != img.size else img
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(base64.b64encode(data)) <= max_base64_bytes:
                return data
            scale *= 0.85
            quality = max(40, quality - 10)
        raise ValueError(
            f"Image compression failed after 32 iterations: "
            f"base64 size {len(base64.b64encode(data))} > limit {max_base64_bytes}"
        )

    def health_check(self) -> bool:
        """Check whether the agent server (:8080) is up."""
        try:
            response = requests.get(f"{self.base_url}/v1/sandbox", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_feedback(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Get feedback from executing an action."""
        raise NotImplementedError("Not implemented")

    def send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON request to the agent server."""
        def _request():
            response = requests.post(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            return validate_response(response)

        return retry_request(_request)

    def create_environment(self, task: Dict[str, Any], wait_time: int = 60) -> bool:
        """Create and start a sandbox environment using the configured runtime."""
        self.runtime_metadata = {
            "type": self.runtime_type,
            "base_url": self.base_url,
            "surfaces": ["shell", "file", "code", "jupyter", "computer-use"],
        }
        return self.runtime.start(task, wait_time)

    def create_docker_environment(self, task: Dict[str, Any], wait_time: int = 60) -> bool:
        """Backward-compatible alias for create_environment."""
        return self.create_environment(task, wait_time)

    def copy_to_container(self, host_path: str, container_path: str) -> bool:
        """Backward-compatible host-to-sandbox copy helper."""
        return self.runtime.copy_to_runtime(host_path, container_path)

    def cleanup_environment(self) -> bool:
        """Clean up the active sandbox environment."""
        return self.runtime.cleanup()

    def cleanup_docker_environment(self) -> bool:
        """Backward-compatible alias for cleanup_environment."""
        return self.cleanup_environment()


class ComputerUseSandboxClient(SandboxClient):
    """Client for UDA's pixel-level GUI control.

    Wraps ``POST /v1/computer-use/action`` (Anthropic Computer Tool,
    Action_20251124 vocabulary). Single endpoint, single response shape
    (``{"output", "error", "base64_image"}``); the agent picks which of
    the 17 actions to invoke.

    Replaces cocoa_env.BrowserSandboxClient (CDP-based, web-only).
    """

    # Map this client's ``action_type`` -> Anthropic action name.
    # action_type is the tool name verbatim (e.g. ``computer_use_left_click``);
    # the Anthropic server expects the prefix stripped.
    _SUPPORTED_ACTIONS = frozenset({
        "screenshot",
        "cursor_position",
        "mouse_move",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "key",
        "type",
        "hold_key",
        "scroll",
        "wait",
        "zoom",
        # 2 client-side extensions ported from anthropic-quickstarts/
        # computer-use-best-practices. uda-desktop's `/v1/computer-use/action`
        # may or may not implement these natively; ``get_feedback`` falls back
        # to xclip via /v1/shell when the server rejects them.
        "read_clipboard",
        "write_clipboard",
    })

    # Allowed payload keys per Anthropic ActionRequest schema (server.py).
    _PAYLOAD_KEYS = frozenset({
        "text",
        "coordinate",
        "start_coordinate",
        "scroll_direction",
        "scroll_amount",
        "duration",
        "key",
        "region",
    })

    def __init__(self, sandbox_config: Dict[str, Any] | None = None, **kwargs):
        super().__init__(sandbox_config, **kwargs)
        self.execution_history: list[Dict[str, Any]] = []

    @classmethod
    def is_computer_use_action(cls, action: Dict[str, Any]) -> bool:
        """True if ``action`` should be dispatched here."""
        if not isinstance(action, dict):
            return False
        action_type = action.get("action_type", "")
        return isinstance(action_type, str) and action_type.startswith(COMPUTER_USE_ACTION_PREFIX)

    def get_feedback(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a single computer-use action."""
        try:
            feedback = self._invoke(action)
        except Exception as e:
            logger.exception("Error executing computer-use action")
            feedback = {"done": False, "message": f"Error: {e}"}
        self.execution_history.append({"action": action, "feedback": feedback})
        return feedback

    def take_screenshot(self) -> tuple[str, str]:
        """Convenience: return ``(base64, status)`` for a fresh screenshot.

        Used by the TaskExecutor loop to capture state after a GUI action.
        """
        feedback = self._invoke({"action_type": "computer_use_screenshot"})
        b64 = feedback.get("image_base64") or ""
        if b64:
            return b64, f"Screenshot taken successfully ({len(b64)} chars base64)"
        return "", feedback.get("message", "Failed to take screenshot")

    def _invoke(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action.get("action_type", "")
        if not isinstance(action_type, str) or not action_type.startswith(COMPUTER_USE_ACTION_PREFIX):
            raise ValueError(
                f"ComputerUseSandboxClient: action_type {action_type!r} is not a computer_use_* action"
            )
        anthropic_action = action_type[len(COMPUTER_USE_ACTION_PREFIX):]
        if anthropic_action not in self._SUPPORTED_ACTIONS:
            raise ValueError(
                f"Unsupported computer-use action: {anthropic_action!r}. "
                f"Supported: {sorted(self._SUPPORTED_ACTIONS)}"
            )

        payload: Dict[str, Any] = {"action": anthropic_action}
        for key in self._PAYLOAD_KEYS:
            if key in action and action[key] is not None:
                payload[key] = action[key]

        result = self.send_request("/v1/computer-use/action", payload)
        return self._normalize_tool_result(action_type, result)

    def _normalize_tool_result(self, action_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Translate Anthropic ToolResult -> internal feedback dict.

        Anthropic shape: ``{output, error, base64_image}``.
        Internal feedback: ``{done, message, image_base64?}``.
        """
        output = result.get("output") or ""
        error = result.get("error") or ""
        image_b64 = result.get("base64_image") or ""

        if error:
            message = f"computer-use {action_type}: error: {error}"
        elif output:
            message = f"computer-use {action_type}: {output.strip()}"
        elif image_b64:
            message = f"computer-use {action_type}: ok ({len(image_b64)} chars base64 image)"
        else:
            message = f"computer-use {action_type}: ok"

        feedback: Dict[str, Any] = {"done": False, "message": message}
        if image_b64:
            # Optional Claude-friendly resize/recompress.
            try:
                raw = base64.b64decode(image_b64)
                compressed = self._compress_image_bytes_for_claude(raw)
                image_b64 = base64.b64encode(compressed).decode("ascii")
            except Exception:
                pass
            feedback["image_base64"] = image_b64
        return feedback

    def get_history(self) -> list[Dict[str, Any]]:
        return self.execution_history

    def clear_history(self) -> None:
        logger.debug(f"Clearing computer-use history ({len(self.execution_history)} entries)")
        self.execution_history = []


class UnifiedSandboxClient(SandboxClient):
    """Single client that dispatches across all UDA surfaces.

    Routes by ``action_type``:

    * ``computer_use_*`` -> :class:`ComputerUseSandboxClient` (GUI control)
    * ``editor`` (view / create / str_replace / insert / undo_edit) ->
      agent-infra/sandbox file API. ``view`` on image extensions returns
      the file as base64 (subsumes legacy ``image_read``).
    * ``python`` (``{code: str}``) -> sandbox code API
    * ``bash`` (``{command: str}``) -> sandbox shell
    * ``task_complete`` / ``exit`` -> terminates the rollout loop
    """

    def __init__(self, sandbox_config: Dict[str, Any] | None = None, **kwargs):
        super().__init__(sandbox_config, **kwargs)
        self.execution_history: list[Dict[str, Any]] = []
        self.sdk_client: Optional[Sandbox] = None
        self._computer_use: Optional[ComputerUseSandboxClient] = None

        self.shell_session_id: Optional[str] = None
        self.jupyter_session_id: Optional[str] = None

    def _initialize_sdk_client(self) -> None:
        """Initialise the agent-infra/sandbox SDK client and computer-use sub-client."""
        if self.sdk_client is None:
            self.sdk_client = Sandbox(base_url=self.base_url)
            logger.debug(f"Initialized Sandbox SDK client with base_url: {self.base_url}")

            try:
                session = self.sdk_client.shell.create_session(exec_dir=UDA_WORKSPACE)
                self.shell_session_id = session.data.session_id
                logger.debug(f"Created shell session: {self.shell_session_id}")
            except Exception as e:
                logger.warning(f"Failed to create shell session: {e}")

            try:
                session = self.sdk_client.jupyter.create_session(kernel_name="python3")
                self.jupyter_session_id = session.data.session_id
                logger.debug(f"Created Jupyter session: {self.jupyter_session_id}")
            except Exception as e:
                logger.warning(f"Failed to create Jupyter session: {e}")

        if self._computer_use is None:
            # Share the base_url + LLM identification; the computer-use
            # client is otherwise standalone (single-endpoint HTTP).
            cu = ComputerUseSandboxClient.__new__(ComputerUseSandboxClient)
            cu.sandbox_config = self.sandbox_config
            cu.port = self.port
            cu.base_url = self.base_url
            cu.container_id = None
            cu.runtime_id = None
            cu.task_name = None
            cu.task_dir = None
            cu.llm_provider = self.llm_provider
            cu.llm_model = self.llm_model
            cu.runtime_type = self.runtime_type
            cu.runtime_metadata = dict(self.runtime_metadata)
            cu.runtime = None  # No lifecycle ownership; UnifiedSandboxClient owns the runtime.
            cu.execution_history = []
            self._computer_use = cu

    def _reset_sdk_client_state(self) -> None:
        """Drop cached SDK state when the backing sandbox changes."""
        self.sdk_client = None
        self.shell_session_id = None
        self.jupyter_session_id = None
        if self._computer_use is not None:
            self._computer_use.clear_history()
        self._computer_use = None

    def get_feedback(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Route ``action`` to the matching surface handler."""
        self._initialize_sdk_client()

        action_type = action.get("action_type")

        if action_type == "task_complete" or action_type == "exit":
            logger.debug("Task completed")
            result_text = action.get("result")
            if result_text:
                logger.debug(f"Task completed with result: {result_text[:200]}...")
                feedback: Dict[str, Any] = {
                    "done": True,
                    "message": f"Task completed. Result: {result_text}",
                    "task_result": result_text,
                }
            else:
                feedback = {"done": True, "message": "Task completed"}
            self.execution_history.append({"action": action, "feedback": feedback})
            return feedback

        try:
            if isinstance(action_type, str) and action_type.startswith(COMPUTER_USE_ACTION_PREFIX):
                return self._handle_computer_use_action(action)

            if action_type == "editor":
                return self._handle_file_action(action)

            if action_type == "python":
                return self._handle_code_action(action)

            if action_type == "bash" or action.get("command"):
                return self._handle_shell_action(action)

            message = f"Unknown action type: {action_type}"
            feedback = {"done": False, "message": message}
            self.execution_history.append({"action": action, "feedback": feedback})
            return feedback

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            feedback = {"done": False, "message": f"Error: {e}"}
            self.execution_history.append({"action": action, "feedback": feedback})
            return feedback

    def take_screenshot(self) -> tuple[str, str]:
        """Take a screenshot via the computer-use surface."""
        self._initialize_sdk_client()
        assert self._computer_use is not None
        return self._computer_use.take_screenshot()

    def _handle_computer_use_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle computer-use (pixel-level GUI) actions."""
        assert self._computer_use is not None
        feedback = self._computer_use.get_feedback(action)
        self.execution_history.extend(self._computer_use.execution_history[-1:])
        return feedback

    _IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

    def _handle_file_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ``editor`` actions (view / create / str_replace / insert / undo_edit).

        Replaces the legacy 8-tool file family. ``view`` on an image file
        downloads it and returns base64 (subsumes the old ``image_read``).
        """
        from agent_sandbox.file.types import Command

        try:
            command = action.get("command")
            path = action.get("path")
            if not command:
                raise ValueError("editor requires 'command' parameter")
            if not path:
                raise ValueError("editor requires 'path' parameter")

            # ``view`` on an image file: download bytes and return as image
            # block. (Subsumes the legacy image_read tool.)
            if command == "view" and isinstance(path, str) and path.lower().endswith(self._IMAGE_EXTENSIONS):
                image_data = b""
                for chunk in self.sdk_client.file.download_file(path=path):
                    image_data += chunk
                if not image_data:
                    raise ValueError(f"Failed to read image file: {path} or file is empty")
                image_data = self._compress_image_bytes_for_claude(image_data)
                base64_image = base64.b64encode(image_data).decode("utf-8")
                message = f"Read image from {path} ({len(image_data)} bytes)"
                feedback = {"done": False, "message": message, "image_base64": base64_image}
                self.execution_history.append({"action": action, "feedback": feedback})
                logger.debug(
                    f"Feedback (OBSERVATION): \n"
                    f"{colorize(json.dumps({**feedback, 'image_base64': f'<{len(base64_image)} chars>'}, indent=2), 'YELLOW')}"
                )
                return feedback

            # All other commands go through the str_replace_editor SDK.
            command_map = {
                "view": Command.VIEW,
                "create": Command.CREATE,
                "str_replace": Command.STR_REPLACE,
                "insert": Command.INSERT,
                "undo_edit": Command.UNDO_EDIT,
            }
            if command not in command_map:
                raise ValueError(
                    f"editor: invalid command '{command}'. Valid: {list(command_map)}"
                )
            kwargs: Dict[str, Any] = {"command": command_map[command], "path": path}
            for opt in ("file_text", "old_str", "new_str", "insert_line", "view_range"):
                if action.get(opt) is not None:
                    kwargs[opt] = action.get(opt)
            result = self.sdk_client.file.str_replace_editor(**kwargs)
            # The SDK call returns the command-specific output (file content
            # for view, ack for create/str_replace/insert/undo_edit). Surface
            # what we have.
            output = ""
            if result is not None and hasattr(result, "data"):
                data = result.data
                output = (
                    getattr(data, "content", None)
                    or getattr(data, "output", None)
                    or ""
                )
            if command == "view":
                if not isinstance(output, str):
                    output = str(output)
                if len(output) > 5000:
                    message = f"{path} (first 5000 chars):\n{output[:5000]}\n... (truncated, total {len(output)} chars)"
                else:
                    message = f"{path}:\n{output}" if output else f"editor view {path}: (empty)"
            else:
                message = f"editor {command} on {path}: ok"

            feedback = {"done": False, "message": message}
            self.execution_history.append({"action": action, "feedback": feedback})
            logger.debug(f"Feedback (OBSERVATION): \n{colorize(json.dumps(feedback, indent=2), 'YELLOW')}")
            return feedback

        except Exception as e:
            logger.error(f"Error executing file action: {e}")
            logger.exception("Full traceback:")
            feedback = {"done": False, "message": f"Error: {e}"}
            self.execution_history.append({"action": action, "feedback": feedback})
            return feedback

    def _handle_code_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ``python`` action (rename of legacy code_execute).

        Python-only; the legacy ``language`` / ``timeout`` parameters were
        dropped to mirror anthropic-quickstarts' PythonTool. If you need
        another runtime, use ``bash`` to invoke it.
        """
        try:
            code = action.get("code")
            if not code:
                raise ValueError("python requires 'code' parameter")

            result = self.sdk_client.code.execute_code(
                language="python", code=code
            )

            data = result.data
            parts = []
            if data.stdout:
                parts.append(data.stdout.rstrip())
            if data.stderr:
                parts.append(f"[stderr]\n{data.stderr.rstrip()}")
            if data.outputs:
                try:
                    parts.append(json.dumps(data.outputs, indent=2))
                except Exception:
                    parts.append(str(data.outputs))

            message = "\n".join([p for p in parts if p]) or f"Code executed: status={data.status}"

            feedback = {"done": False, "message": message}
            self.execution_history.append({"action": action, "feedback": feedback})
            logger.debug(f"Feedback (OBSERVATION): \n{colorize(json.dumps(feedback, indent=2), 'YELLOW')}")
            return feedback

        except Exception as e:
            logger.error(f"Error executing code action: {e}")
            logger.exception("Full traceback:")
            feedback = {"done": False, "message": f"Error: {e}"}
            self.execution_history.append({"action": action, "feedback": feedback})
            return feedback

    def _handle_shell_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shell command execution."""
        try:
            command = action.get("command")
            if not command:
                raise ValueError("bash requires 'command' parameter")

            if not self.shell_session_id:
                try:
                    session = self.sdk_client.shell.create_session(exec_dir=UDA_WORKSPACE)
                    self.shell_session_id = session.data.session_id
                    logger.debug(f"Created new shell session: {self.shell_session_id}")
                except Exception as e:
                    logger.warning(f"Failed to create shell session, will let SDK auto-create: {e}")

            try:
                result = self.sdk_client.shell.exec_command(
                    command=command,
                    id=self.shell_session_id,
                    exec_dir=UDA_WORKSPACE,
                    async_mode=False,
                    timeout=0,
                )
                if hasattr(result, "data") and hasattr(result.data, "session_id"):
                    self.shell_session_id = result.data.session_id

                output = result.data.output
                message = output if output else "Command executed successfully (no output)"
            except Exception as session_error:
                error_str = str(session_error)
                if "Session not found" in error_str or "404" in error_str:
                    logger.warning(f"Session {self.shell_session_id} not found, creating new session")
                    session = self.sdk_client.shell.create_session(exec_dir=UDA_WORKSPACE)
                    self.shell_session_id = session.data.session_id
                    logger.debug(f"Created new shell session after error: {self.shell_session_id}")

                    result = self.sdk_client.shell.exec_command(
                        command=command,
                        id=self.shell_session_id,
                        exec_dir=UDA_WORKSPACE,
                        async_mode=False,
                        timeout=0,
                    )
                    output = result.data.output
                    message = output if output else "Command executed successfully (no output)"
                else:
                    raise

            feedback = {"done": False, "message": message}
            self.execution_history.append({"action": action, "feedback": feedback})
            logger.debug(f"Feedback (OBSERVATION): \n{colorize(json.dumps(feedback, indent=2), 'YELLOW')}")
            return feedback

        except Exception as e:
            logger.error(f"Error executing shell action: {e}")
            logger.exception("Full traceback:")
            feedback = {"done": False, "message": f"Error: {e}"}
            self.execution_history.append({"action": action, "feedback": feedback})
            return feedback

    def get_history(self) -> list[Dict[str, Any]]:
        """Get the recorded execution history."""
        return self.execution_history

    def clear_history(self) -> None:
        """Clear the execution history."""
        logger.debug(f"Clearing execution history ({len(self.execution_history)} entries)")
        self.execution_history = []
        if self._computer_use is not None:
            self._computer_use.clear_history()

    def create_environment(self, task: Dict[str, Any], wait_time: int = 60) -> bool:
        """Create and initialise the configured sandbox runtime."""
        if not super().create_environment(task, wait_time):
            return False
        self.clear_history()
        self._reset_sdk_client_state()
        return True
