"""Harbor-style Qwen Code agent adapted for NanoRollout."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from nanorollout.envs.shell_env.base import ShellEnvironment

from .installed_agent import EnvVar, InstalledAgentBase


class QwenCode(InstalledAgentBase):
    """Qwen Code CLI agent."""

    ENV_VARS = [
        EnvVar(
            "api_key", env="OPENAI_API_KEY", type="str", env_fallback="OPENAI_API_KEY"
        ),
        EnvVar(
            "base_url",
            env="OPENAI_BASE_URL",
            type="str",
            env_fallback="OPENAI_BASE_URL",
        ),
    ]

    @staticmethod
    def name() -> str:
        return "qwen-code"

    def get_version_command(self) -> str | None:
        return ". ~/.nvm/nvm.sh; qwen --version"

    def install(self, environment: ShellEnvironment) -> None:
        self.exec(
            environment,
            "apt-get update && apt-get install -y curl",
            env={"DEBIAN_FRONTEND": "noninteractive"},
            timeout_sec=self.install_timeout_sec,
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        self.exec(
            environment,
            (
                "set -euo pipefail; "
                "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                'export NVM_DIR="$HOME/.nvm" && '
                '\\. "$NVM_DIR/nvm.sh" || true && '
                "command -v nvm >/dev/null 2>&1 || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
                "nvm install 22 && npm -v && "
                f"npm install -g @qwen-code/qwen-code{version_spec} && "
                "qwen --version"
            ),
            timeout_sec=self.install_timeout_sec,
        )

    def _build_register_skills_command(self) -> str | None:
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p ~/.qwen/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            "~/.qwen/skills/ 2>/dev/null || true"
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        if not self.mcp_servers:
            return None
        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if server.transport == "stdio":
                servers[server.name] = {"command": server.command, "args": server.args}
            elif server.transport == "streamable-http":
                servers[server.name] = {"httpUrl": server.url}
            else:
                servers[server.name] = {"url": server.url}
        config = json.dumps({"mcpServers": servers}, indent=2)
        return (
            "mkdir -p ~/.qwen && " f"cat > ~/.qwen/settings.json <<'EOF'\n{config}\nEOF"
        )

    def invoke(
        self,
        instruction: str,
        environment: ShellEnvironment,
        *,
        timeout_sec: int | None = None,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)
        env = self.resolve_env_vars()
        if self.model_name:
            env["OPENAI_MODEL"] = self.model_name
        elif self._has_env("OPENAI_MODEL"):
            env["OPENAI_MODEL"] = self._get_env("OPENAI_MODEL") or ""
        else:
            env["OPENAI_MODEL"] = "qwen3-coder-plus"

        skills_command = self._build_register_skills_command()
        if skills_command:
            self.exec(environment, skills_command, env=env, timeout_sec=120)

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            self.exec(environment, mcp_command, env=env, timeout_sec=120)

        output_path = self.remote_logs_dir / "qwen-code.txt"
        sessions_dir = self.remote_logs_dir / "qwen-sessions"
        try:
            self.exec(
                environment,
                (
                    ". ~/.nvm/nvm.sh; "
                    f"qwen --yolo --prompt={escaped_instruction} "
                    f"2>&1 | stdbuf -oL tee {shlex.quote(output_path.as_posix())}"
                ),
                env=env,
                timeout_sec=timeout_sec,
            )
        finally:
            self.exec(
                environment,
                (
                    f"mkdir -p {shlex.quote(sessions_dir.as_posix())} && "
                    "cp -r ~/.qwen/projects/. "
                    f"{shlex.quote(sessions_dir.as_posix())}/ 2>/dev/null || true"
                ),
                timeout_sec=60,
                check=False,
            )

    def _find_session_jsonl(self) -> Path | None:
        sessions_dir = self.logs_dir / "qwen-sessions"
        if not sessions_dir.is_dir():
            return None
        jsonl_files = list(sessions_dir.rglob("*.jsonl"))
        if not jsonl_files:
            return None
        return max(jsonl_files, key=lambda path: path.stat().st_mtime)

    def _parse_jsonl(self) -> list[dict[str, Any]]:
        path = self._find_session_jsonl()
        if path is None:
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
        return events

    @staticmethod
    def _make_tool_call(
        call_id: str, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "tool_call_id": call_id,
            "function_name": name,
            "arguments": arguments,
        }

    @staticmethod
    def _make_metrics(
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        thoughts_tokens: int,
    ) -> dict[str, Any] | None:
        if not prompt_tokens and not completion_tokens:
            return None
        extra = {"thoughts_tokens": thoughts_tokens} if thoughts_tokens else None
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens or None,
            "cost_usd": None,
            "extra": extra,
        }

    def build_trajectory(self) -> dict[str, Any] | None:
        events = self._parse_jsonl()
        if not events:
            return None

        session_id = "unknown"
        agent_version = self.version() or "unknown"
        model_name = self.model_name
        for event in events:
            if event.get("sessionId"):
                session_id = event["sessionId"]
            if event.get("version"):
                agent_version = event["version"]
            if event.get("model"):
                model_name = event["model"]

        steps: list[dict[str, Any]] = []
        step_id = 1
        total_prompt = 0
        total_completion = 0
        total_cached = 0

        for event in events:
            event_type = event.get("type")
            message = event.get("message")
            if not isinstance(message, dict):
                continue

            if event_type == "user":
                parts = message.get("parts", [])
                text = "\n".join(
                    part.get("text", "") for part in parts if part.get("text")
                )
                if not text:
                    continue
                steps.append(
                    {
                        "step_id": step_id,
                        "timestamp": event.get("timestamp"),
                        "source": "user",
                        "message": text,
                    }
                )
                step_id += 1
                continue

            if event_type == "assistant":
                parts = message.get("parts", [])
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for part in parts:
                    if part.get("text"):
                        text_parts.append(part["text"])
                    if part.get("functionCall"):
                        call = part["functionCall"]
                        tool_calls.append(
                            self._make_tool_call(
                                call.get("id") or "",
                                call.get("name") or "",
                                call.get("args") or {},
                            )
                        )

                usage = event.get("usageMetadata") or {}
                prompt_tokens = usage.get("promptTokenCount", 0) or 0
                completion_tokens = usage.get("candidatesTokenCount", 0) or 0
                cached_tokens = usage.get("cachedContentTokenCount", 0) or 0
                thoughts_tokens = usage.get("thoughtsTokenCount", 0) or 0

                total_prompt += prompt_tokens
                total_completion += completion_tokens
                total_cached += cached_tokens

                step: dict[str, Any] = {
                    "step_id": step_id,
                    "timestamp": event.get("timestamp"),
                    "source": "agent",
                    "message": "\n".join(text_parts) or "(tool use)",
                    "model_name": model_name,
                }
                if tool_calls:
                    step["tool_calls"] = tool_calls
                metrics = self._make_metrics(
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    thoughts_tokens,
                )
                if metrics:
                    step["metrics"] = metrics
                steps.append(step)
                step_id += 1
                continue

            if event_type == "tool_result":
                parts = message.get("parts", [])
                for part in parts:
                    function_response = part.get("functionResponse")
                    if not function_response:
                        continue
                    call_id = function_response.get("id") or ""
                    output = (function_response.get("response") or {}).get("output", "")
                    for step in reversed(steps):
                        if step.get("source") != "agent" or not step.get("tool_calls"):
                            continue
                        if any(
                            tool_call.get("tool_call_id") == call_id
                            for tool_call in step["tool_calls"]
                        ):
                            result = {
                                "source_call_id": call_id,
                                "content": str(output),
                            }
                            observation = step.setdefault(
                                "observation", {"results": []}
                            )
                            observation["results"].append(result)
                            break

        if not steps:
            return None

        return {
            "session_id": session_id,
            "agent": {
                "name": "qwen-code",
                "version": agent_version,
                "model_name": model_name,
            },
            "steps": steps,
            "final_metrics": {
                "total_prompt_tokens": total_prompt or None,
                "total_completion_tokens": total_completion or None,
                "total_cached_tokens": total_cached or None,
                "total_cost_usd": None,
                "total_steps": len(steps),
            },
        }
