"""Modal-backed sandbox runtime for UDA tasks.

Constructs a ``modal.Image.from_dockerfile`` against the per-task
``Dockerfile`` (already FROM uda-desktop after migration), spawns a
``modal.Sandbox`` with ``encrypted_ports=[8080]``, and exposes the
sandbox's tunnel URL for the agent loop. Identical to
cocoa_env.modal at the runtime layer.
"""

from __future__ import annotations

import time
import modal

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseSandboxRuntime, runtime_logger


class ModalSandboxRuntime(BaseSandboxRuntime):
    """Lifecycle manager backed by Modal sandboxes."""

    runtime_type = "modal"

    def __init__(self, client):
        super().__init__(client)
        self.sandbox: Optional[Any] = None
        self.app: Optional[Any] = None
        self.service_port = 8080

    def start(self, task: Dict[str, Any], wait_time: int = 60) -> bool:
        try:
            task_dir = task.get("task_dir")
            task_name = task.get("task_name", "task")
            if not task_dir:
                runtime_logger.error("Task object must contain 'task_dir' key")
                return False

            task_path = Path(task_dir)
            dockerfile_path = task_path / "Dockerfile"
            if not dockerfile_path.exists():
                runtime_logger.error("Task '%s' is missing Dockerfile at %s", task_name, dockerfile_path)
                return False

            self.client.task_name = task_name
            self.client.task_dir = task_dir
            self.service_port = int(self.client.sandbox_config.get("modal_container_port", 8080))

            app_name = self.client.sandbox_config.get("modal_app_name", "__nanorollout_uda__")
            bench = (self.client.sandbox_config.get("bench") or "uda").strip() or "uda"
            sandbox_name = (
                self.client.sandbox_config.get("modal_sandbox_name")
                or f"uda-{bench}-{task_name}-{int(time.time())}"
            )
            startup_timeout = int(self.client.sandbox_config.get("modal_startup_timeout", 300))
            sandbox_timeout = int(self.client.sandbox_config.get("modal_timeout", 3600))
            idle_timeout = self.client.sandbox_config.get("modal_idle_timeout", 600)

            self.app = modal.App.lookup(app_name, create_if_missing=True)
            image = modal.Image.from_dockerfile(
                str(dockerfile_path.resolve()),
                context_dir=str(task_path.resolve()),
            )

            create_kwargs: Dict[str, Any] = {
                "app": self.app,
                "image": image,
                "encrypted_ports": [self.service_port],
                "timeout": sandbox_timeout,
                "name": sandbox_name,
            }
            if idle_timeout is not None:
                create_kwargs["idle_timeout"] = int(idle_timeout)

            region = self.client.sandbox_config.get("modal_region")
            if region:
                create_kwargs["region"] = region

            cpu = self.client.sandbox_config.get("modal_cpu", 1)
            if cpu is not None:
                create_kwargs["cpu"] = cpu

            memory = self.client.sandbox_config.get("modal_memory", 2048)
            if memory is not None:
                create_kwargs["memory"] = memory

            runtime_logger.info(
                "Starting Modal sandbox for task '%s' (app=%s, dockerfile=%s)",
                task_name,
                app_name,
                dockerfile_path,
            )
            self.sandbox = modal.Sandbox.create(**create_kwargs)

            tunnel = self.sandbox.tunnels()[self.service_port]
            self.client.set_base_url(tunnel.url)
            self.client.runtime_id = getattr(self.sandbox, "object_id", None)
            self.client.container_id = self.client.runtime_id
            self.client._update_runtime_metadata(
                sandbox_id=self.client.runtime_id,
                app_name=app_name,
                sandbox_name=sandbox_name,
                task_name=task_name,
                task_dir=task_dir,
                container_port=self.service_port,
                bench=bench,
                uda_image=self.client.sandbox_config.get("uda_image"),
                corpus_revision=self.client.sandbox_config.get("corpus_revision"),
            )

            health_timeout = max(wait_time, startup_timeout)
            if self._wait_for_health(health_timeout):
                runtime_logger.info("Modal sandbox environment ready")
                return True

            runtime_logger.error(
                "Modal sandbox environment failed to become ready within timeout of %s seconds",
                health_timeout,
            )
            self.cleanup()
            return False
        except Exception as e:
            runtime_logger.error("Error creating Modal sandbox: %s", e)
            self.cleanup()
            return False

    def cleanup(self) -> bool:
        if self.sandbox is None:
            runtime_logger.info("No Modal sandbox to clean up")
            return True

        try:
            sandbox_id = getattr(self.sandbox, "object_id", None)
            runtime_logger.info("Terminating Modal sandbox %s", sandbox_id or "<unknown>")
            self.sandbox.terminate()
            self.client.container_id = None
            self.client.runtime_id = None
            return True
        except Exception as e:
            runtime_logger.error("Error terminating Modal sandbox: %s", e)
            return False
        finally:
            self.sandbox = None

    # copy_to_runtime + exec_in_runtime inherit the SDK-mediated default
    # impl from BaseSandboxRuntime, which talks HTTP to the sandbox
    # server through ``client.sdk_client``. The Modal tunnel URL is
    # exposed at ``client.set_base_url(tunnel.url)`` so the SDK calls
    # land on the right endpoint inside the modal container.
