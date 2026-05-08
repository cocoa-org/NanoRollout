"""Modal-backed sandbox runtime for Cocoa tasks."""

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

            app_name = self.client.sandbox_config.get("modal_app_name", "__nanorollout__")
            sandbox_name = (
                self.client.sandbox_config.get("modal_sandbox_name")
                or f"cocoa-task-{task_name}-{int(time.time())}"
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

    def copy_to_runtime(self, host_path: str, container_path: str) -> bool:
        runtime_logger.error("copy_to_container is not implemented for the Modal runtime")
        return False
