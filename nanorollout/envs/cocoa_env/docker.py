"""Docker-backed sandbox runtime for Cocoa tasks."""

import subprocess

from pathlib import Path
from typing import Any, Dict

from .base import BaseSandboxRuntime, runtime_logger


class DockerComposeSandboxRuntime(BaseSandboxRuntime):
    """Lifecycle manager backed by local docker compose."""

    runtime_type = "docker"

    def start(self, task: Dict[str, Any], wait_time: int = 60) -> bool:
        try:
            task_dir = task.get("task_dir")
            task_name = task.get("task_name", "task")

            if not task_dir:
                runtime_logger.error("Task object must contain 'task_dir' key")
                return False

            self.client.task_name = task_name
            self.client.task_dir = task_dir
            docker_compose_path = f"{task_dir}/docker-compose.yaml"

            env = {
                "TASK_DOCKER_IMAGE_NAME": f"task-{task_name}:latest",
                "TASK_DOCKER_CONTAINER_NAME": f"task-{task_name}-container",
                "HOST_PORT": str(self.client.port),
            }

            runtime_logger.info(
                "Building and starting sandbox for task '%s' using docker-compose",
                task_name,
            )

            build_result = subprocess.run(
                ["docker", "compose", "-f", docker_compose_path, "build", "--no-cache"],
                capture_output=True,
                text=True,
                timeout=120,
                env={**subprocess.os.environ, **env},
            )
            if build_result.returncode != 0:
                runtime_logger.error("Failed to build container with docker-compose: %s", build_result.stderr)
                return False

            result = subprocess.run(
                ["docker", "compose", "-f", docker_compose_path, "up", "-d"],
                capture_output=True,
                text=True,
                timeout=120,
                env={**subprocess.os.environ, **env},
            )
            if result.returncode != 0:
                runtime_logger.error("Failed to start container with docker-compose: %s", result.stderr)
                return False

            self.client.container_id = env["TASK_DOCKER_CONTAINER_NAME"]
            self.client.runtime_id = self.client.container_id
            self.client._update_runtime_metadata(
                container_id=self.client.container_id,
                docker_port=self.client.port,
                task_name=task_name,
                task_dir=task_dir,
            )

            if self._wait_for_health(wait_time):
                runtime_logger.info("Docker sandbox environment ready")
                return True

            runtime_logger.error(
                "Docker sandbox environment failed to become ready within timeout of %s seconds",
                wait_time,
            )
            return False
        except subprocess.TimeoutExpired:
            runtime_logger.error("Docker command timed out")
            return False
        except Exception as e:
            runtime_logger.error("Error creating agent server: %s", e)
            return False

    def cleanup(self) -> bool:
        try:
            if self.client.task_dir and self.client.task_name:
                docker_compose_path = f"{self.client.task_dir}/docker-compose.yaml"
                env = {
                    "TASK_DOCKER_IMAGE_NAME": f"task-{self.client.task_name}:latest",
                    "TASK_DOCKER_CONTAINER_NAME": f"task-{self.client.task_name}-container",
                    "HOST_PORT": str(self.client.port),
                }
                runtime_logger.info(
                    "Stopping sandbox for task '%s' using docker-compose",
                    self.client.task_name,
                )

                result = subprocess.run(
                    ["docker", "compose", "-f", docker_compose_path, "down"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={**subprocess.os.environ, **env},
                )

                if result.returncode != 0:
                    runtime_logger.error("Failed to stop container: %s", result.stderr)
                    return False

                runtime_logger.info("Agent server container stopped successfully")
                self.client.container_id = None
                self.client.runtime_id = None
                return True

            runtime_logger.info("No container to clean up")
            return True
        except subprocess.TimeoutExpired:
            runtime_logger.error("Docker command timed out")
            return False
        except Exception as e:
            runtime_logger.error("Error cleaning up agent server: %s", e)
            return False

    def copy_to_runtime(self, host_path: str, container_path: str) -> bool:
        try:
            if not self.client.container_id:
                runtime_logger.error("No container running. Call create_environment first.")
                return False

            host_file = Path(host_path)
            if not host_file.exists():
                runtime_logger.error("Source path does not exist: %s", host_path)
                return False

            parent_dir = str(Path(container_path).parent)
            if parent_dir and parent_dir != "/":
                mkdir_result = subprocess.run(
                    ["docker", "exec", self.client.container_id, "mkdir", "-p", parent_dir],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if mkdir_result.returncode != 0:
                    runtime_logger.error("Failed to create parent directory: %s", mkdir_result.stderr)
                    return False

            runtime_logger.info(
                "Copying %s to container %s:%s",
                host_path,
                self.client.container_id,
                container_path,
            )
            result = subprocess.run(
                ["docker", "cp", str(host_file), f"{self.client.container_id}:{container_path}"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                runtime_logger.info("Successfully copied %s to container", host_path)
                return True

            runtime_logger.error("Failed to copy file to container: %s", result.stderr)
            return False
        except subprocess.TimeoutExpired:
            runtime_logger.error("Docker copy command timed out")
            return False
        except Exception as e:
            runtime_logger.error("Error copying file to container: %s", e)
            return False
