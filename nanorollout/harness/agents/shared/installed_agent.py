"""Shared support for Harbor-style installed CLI agents."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shlex
import tarfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal

from nanorollout.envs.shell_env.base import ExecutionResult, ShellEnvironment


DEFAULT_INSTALL_TIMEOUT_SEC = 1800
DEFAULT_VERSION_TIMEOUT_SEC = 60
DEFAULT_SYNC_TIMEOUT_SEC = 120

EXIT_STATUS_FINISHED = "finished"
EXIT_STATUS_ERROR = "error"

logger = logging.getLogger(__name__)


class NonZeroAgentExitCodeError(RuntimeError):
    """Raised when an installed agent command exits non-zero."""


def _parse_bool_env_value(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if not isinstance(value, str):
        raise ValueError(f"Invalid value for '{name}': expected bool-like value")

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid value for '{name}': {value!r}")


@dataclass
class CliFlag:
    """Declarative CLI flag definition."""

    kwarg: str
    cli: str
    type: Literal["str", "int", "bool", "enum"] = "str"
    choices: list[str] | None = None
    default: Any = None
    env_fallback: str | None = None
    format: str | None = None


@dataclass
class EnvVar:
    """Declarative environment variable definition."""

    kwarg: str
    env: str
    type: Literal["str", "int", "bool", "enum"] = "str"
    choices: list[str] | None = None
    default: Any = None
    env_fallback: str | None = None
    bool_true: str = "true"
    bool_false: str = "false"


@dataclass
class MCPServer:
    """Minimal MCP server config used by Harbor's installed agents."""

    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None

    @classmethod
    def from_any(cls, value: Any) -> "MCPServer":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"Invalid MCP server config: {value!r}")
        return cls(
            name=str(value["name"]),
            transport=str(value["transport"]),
            command=value.get("command"),
            args=list(value.get("args") or []),
            url=value.get("url"),
        )


@dataclass
class AgentResult:
    """Minimal runner-facing agent result."""

    success: bool
    message: str
    history: list[dict[str, Any]] = field(default_factory=list)
    llm_metrics: list[dict[str, Any]] = field(default_factory=list)
    llm_cost_total: float = 0.0
    patch: str = ""
    iterations: int = 0
    error: str | None = None
    exit_status: str = EXIT_STATUS_FINISHED


def _coerce_value(
    value: Any,
    type_name: Literal["str", "int", "bool", "enum"],
    choices: list[str] | None,
    kwarg_name: str,
) -> Any:
    if type_name == "str":
        if isinstance(value, bool):
            raise ValueError(
                f"Invalid value for '{kwarg_name}': expected str, got bool"
            )
        if isinstance(value, (int, float)):
            return str(value)
        if not isinstance(value, str):
            raise ValueError(
                f"Invalid value for '{kwarg_name}': expected str, got {type(value).__name__}"
            )
        return value

    if type_name == "int":
        if isinstance(value, bool):
            raise ValueError(
                f"Invalid value for '{kwarg_name}': expected int, got bool"
            )
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, str):
            return int(value)
        raise ValueError(
            f"Invalid value for '{kwarg_name}': expected int, got {type(value).__name__}"
        )

    if type_name == "bool":
        return _parse_bool_env_value(value, name=kwarg_name)

    if type_name == "enum":
        if not isinstance(value, str):
            raise ValueError(
                f"Invalid value for '{kwarg_name}': expected str, got {type(value).__name__}"
            )
        normalized = value.strip().lower()
        if choices and normalized not in choices:
            valid = ", ".join(sorted(choices))
            raise ValueError(
                f"Invalid value for '{kwarg_name}': {value!r}. Valid values: {valid}"
            )
        return normalized

    raise ValueError(f"Unknown descriptor type: {type_name}")


class InstalledAgentBase(ABC):
    """Shared NanoRollout port of Harbor's installed-agent contract."""

    CLI_FLAGS: ClassVar[list[CliFlag]] = []
    ENV_VARS: ClassVar[list[EnvVar]] = []

    def __init__(
        self,
        *,
        logs_dir: Path,
        model_name: str | None = None,
        version: str | None = None,
        logger_override: logging.Logger | None = None,
        extra_env: dict[str, str] | None = None,
        skills_dir: str | None = None,
        mcp_servers: list[MCPServer | dict[str, Any]] | None = None,
        install_timeout_sec: int = DEFAULT_INSTALL_TIMEOUT_SEC,
        **kwargs: Any,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name
        self._version = version
        self.skills_dir = skills_dir
        self.install_timeout_sec = install_timeout_sec
        self.logger = logger_override or logger.getChild(self.__class__.__name__)
        self._extra_env = dict(extra_env or {})
        self.mcp_servers = [MCPServer.from_any(value) for value in (mcp_servers or [])]

        self._flag_kwargs: dict[str, Any] = {}
        for descriptor in [*self.CLI_FLAGS, *self.ENV_VARS]:
            if descriptor.kwarg in kwargs:
                self._flag_kwargs[descriptor.kwarg] = kwargs.pop(descriptor.kwarg)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected kwargs for {self.__class__.__name__}: {unexpected}"
            )

        self._resolved_flags = self._resolve_flag_values()
        self._resolved_env_vars = self._resolve_env_values()
        self._remote_logs_dir: PurePosixPath | None = None

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Stable agent name."""

    def version(self) -> str | None:
        return self._version

    def _resolve_raw_value(self, descriptor: CliFlag | EnvVar) -> Any:
        if descriptor.kwarg in self._flag_kwargs:
            return self._flag_kwargs[descriptor.kwarg]
        if descriptor.env_fallback and descriptor.env_fallback in os.environ:
            return os.environ[descriptor.env_fallback]
        return descriptor.default

    def _resolve_flag_values(self) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for descriptor in self.CLI_FLAGS:
            raw = self._resolve_raw_value(descriptor)
            if raw is None:
                continue
            resolved[descriptor.kwarg] = _coerce_value(
                raw, descriptor.type, descriptor.choices, descriptor.kwarg
            )
        return resolved

    def _resolve_env_values(self) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for descriptor in self.ENV_VARS:
            raw = self._resolve_raw_value(descriptor)
            if raw is None:
                continue
            coerced = _coerce_value(
                raw, descriptor.type, descriptor.choices, descriptor.kwarg
            )
            if descriptor.type == "bool":
                resolved[descriptor.env] = (
                    descriptor.bool_true if coerced else descriptor.bool_false
                )
            else:
                resolved[descriptor.env] = str(coerced)
        return resolved

    def resolve_env_vars(self) -> dict[str, str]:
        return dict(self._resolved_env_vars)

    def build_cli_flags(self) -> str:
        parts: list[str] = []
        for descriptor in self.CLI_FLAGS:
            value = self._resolved_flags.get(descriptor.kwarg)
            if value is None:
                continue
            if descriptor.format is not None:
                parts.append(descriptor.format.format(value=value))
            elif descriptor.type == "bool":
                if value:
                    parts.append(descriptor.cli)
            else:
                parts.append(f"{descriptor.cli} {shlex.quote(str(value))}")
        return " ".join(parts)

    def _get_env(self, key: str) -> str | None:
        if key in self._extra_env:
            return self._extra_env[key]
        return os.environ.get(key)

    def _has_env(self, key: str) -> bool:
        return key in self._extra_env or key in os.environ

    def _get_env_prefixed(self, prefix: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                result[key[len(prefix) :]] = value
        for key, value in self._extra_env.items():
            if key.startswith(prefix):
                result[key[len(prefix) :]] = value
        return result

    @property
    def remote_logs_dir(self) -> PurePosixPath:
        if self._remote_logs_dir is None:
            raise RuntimeError("Remote logs dir has not been prepared")
        return self._remote_logs_dir

    def _prepare_remote_logs_dir(self, environment: ShellEnvironment) -> None:
        slug = self.name().replace("_", "-")
        self._remote_logs_dir = PurePosixPath("/tmp") / (
            f"nanorollout-{slug}-{uuid.uuid4().hex[:8]}"
        )
        self.exec(
            environment,
            f"mkdir -p {shlex.quote(self.remote_logs_dir.as_posix())}",
            timeout_sec=30,
        )

    def _build_wrapped_command(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        merged_env = dict(env or {})
        if self._extra_env:
            merged_env.update(self._extra_env)

        parts = ["set -o pipefail"]
        for key, value in merged_env.items():
            if value is None:
                continue
            parts.append(f"export {key}={shlex.quote(str(value))}")
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        parts.append(command)
        return "; ".join(parts)

    def exec(
        self,
        environment: ShellEnvironment,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        check: bool = True,
    ) -> ExecutionResult:
        wrapped = self._build_wrapped_command(command, env=env, cwd=cwd)
        result = environment.execute(wrapped, timeout=timeout_sec)
        if check and result.exit_code != 0:
            raise NonZeroAgentExitCodeError(
                f"Command failed (exit {result.exit_code}): {command}\n{result.output}"
            )
        return result

    def get_version_command(self) -> str | None:
        return None

    def parse_version(self, stdout: str) -> str:
        return stdout.strip()

    def _detect_version(self, environment: ShellEnvironment) -> None:
        if self._version is not None:
            return
        command = self.get_version_command()
        if not command:
            return
        try:
            result = self.exec(
                environment,
                command,
                timeout_sec=DEFAULT_VERSION_TIMEOUT_SEC,
                check=False,
            )
        except Exception:
            return
        if result.exit_code == 0 and result.output.strip():
            self._version = self.parse_version(result.output)

    @staticmethod
    def _safe_extract_tar(data: bytes, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive.getmembers():
                member_path = (destination / member.name).resolve()
                if os.path.commonpath([str(destination), str(member_path)]) != str(
                    destination
                ):
                    raise ValueError(f"Unsafe tar member path: {member.name}")
            archive.extractall(destination)

    def _sync_remote_logs(self, environment: ShellEnvironment) -> None:
        if self._remote_logs_dir is None:
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        remote_dir = shlex.quote(self.remote_logs_dir.as_posix())
        result = self.exec(
            environment,
            (
                f"if [ -d {remote_dir} ]; then "
                f"tar -C {remote_dir} -cf - . | base64 | tr -d '\\n'; "
                "fi"
            ),
            timeout_sec=DEFAULT_SYNC_TIMEOUT_SEC,
            check=False,
        )
        payload = result.output.strip()
        if result.exit_code != 0 or not payload:
            return
        data = base64.b64decode(payload)
        self._safe_extract_tar(data, self.logs_dir)

    def _write_trajectory(self, trajectory: dict[str, Any]) -> None:
        path = self.logs_dir / "trajectory.json"
        path.write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _build_output_trajectory(
        self,
        raw_trajectory: dict[str, Any],
        *,
        history: list[dict[str, Any]],
        llm_metrics: list[dict[str, Any]],
        llm_cost_total: float,
        success: bool,
        message: str,
        iterations: int,
        error: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "success": success,
            "message": message,
            "iterations": iterations,
            "error": error,
            "messages": history,
            "llm_metrics": llm_metrics,
            "llm_cost_total": llm_cost_total,
        }
        session_id = raw_trajectory.get("session_id")
        if session_id is not None:
            payload["session_id"] = session_id
        agent = raw_trajectory.get("agent")
        if isinstance(agent, dict):
            payload["agent"] = agent
        return payload

    def _trajectory_to_history(
        self, trajectory: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not trajectory:
            return []
        messages: list[dict[str, Any]] = []
        for step in trajectory.get("steps", []):
            source = step.get("source")
            if source == "agent":
                role = "assistant"
            elif source == "user":
                role = "user"
            else:
                role = "system"
            message: dict[str, Any] = {
                "role": role,
                "content": step.get("message", ""),
            }
            if role == "assistant" and step.get("reasoning_content"):
                message["reasoning_content"] = step["reasoning_content"]
            if step.get("tool_calls"):
                message["tool_calls"] = step["tool_calls"]
            if step.get("observation") is not None:
                message["observation"] = step["observation"]
            if step.get("model_name"):
                message["model"] = step["model_name"]
            messages.append(message)
        return messages

    def _trajectory_to_llm_metrics(
        self, trajectory: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not trajectory:
            return []
        metrics: list[dict[str, Any]] = []
        for step in trajectory.get("steps", []):
            usage = step.get("metrics")
            if not isinstance(usage, dict):
                continue
            metrics.append(
                {
                    "model": step.get("model_name") or self.model_name,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "cached_tokens": usage.get("cached_tokens"),
                        "extra": usage.get("extra"),
                    },
                    "cost": usage.get("cost_usd"),
                }
            )
        return metrics

    def _result_message(self, history: list[dict[str, Any]], error: str | None) -> str:
        for message in reversed(history):
            if message.get("role") == "assistant":
                content = (message.get("content") or "").strip()
                if content:
                    return content
        return error or ""

    @abstractmethod
    def install(self, environment: ShellEnvironment) -> None:
        """Install the agent into the environment."""

    @abstractmethod
    def invoke(
        self,
        instruction: str,
        environment: ShellEnvironment,
        *,
        timeout_sec: int | None = None,
    ) -> None:
        """Run the agent once against the task."""

    @abstractmethod
    def build_trajectory(self) -> dict[str, Any] | None:
        """Build a local trajectory payload from synced artifacts."""

    def run(
        self,
        instruction: str,
        environment: ShellEnvironment,
        *,
        timeout_sec: int | None = None,
    ) -> AgentResult:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trajectory: dict[str, Any] | None = None
        error: str | None = None
        exit_status = EXIT_STATUS_FINISHED

        try:
            self._prepare_remote_logs_dir(environment)
            self.install(environment)
            self._detect_version(environment)
            self.invoke(instruction, environment, timeout_sec=timeout_sec)
        except Exception as exc:
            self.logger.exception("%s agent run failed", self.name())
            error = str(exc)
            exit_status = EXIT_STATUS_ERROR
        finally:
            try:
                self._sync_remote_logs(environment)
            except Exception:
                self.logger.exception("%s artifact sync failed", self.name())
            try:
                trajectory = self.build_trajectory()
            except Exception:
                self.logger.exception("%s trajectory parsing failed", self.name())

        history = self._trajectory_to_history(trajectory)
        llm_metrics = self._trajectory_to_llm_metrics(trajectory)
        final_metrics = (trajectory or {}).get("final_metrics", {})
        llm_cost_total = final_metrics.get("total_cost_usd")
        if llm_cost_total is None:
            llm_cost_total = sum(
                metric.get("cost", 0.0)
                for metric in llm_metrics
                if isinstance(metric.get("cost"), (int, float))
            )

        patch = environment.get_git_diff()
        message = self._result_message(history, error)
        iterations = sum(1 for item in history if item.get("role") == "assistant")

        if trajectory:
            try:
                output_trajectory = self._build_output_trajectory(
                    trajectory,
                    history=history,
                    llm_metrics=llm_metrics,
                    llm_cost_total=float(llm_cost_total or 0.0),
                    success=error is None,
                    message=message,
                    iterations=iterations,
                    error=error,
                )
                self._write_trajectory(output_trajectory)
            except Exception:
                self.logger.exception("%s trajectory write failed", self.name())

        return AgentResult(
            success=error is None,
            message=message,
            history=history,
            llm_metrics=llm_metrics,
            llm_cost_total=float(llm_cost_total or 0.0),
            patch=patch,
            iterations=iterations,
            error=error,
            exit_status=exit_status,
        )
