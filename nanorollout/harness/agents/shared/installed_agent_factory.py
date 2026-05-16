"""Factory helpers for installed CLI agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .claude_code import ClaudeCode
from .opencode import OpenCode
from .qwen_code import QwenCode

AGENT_REGISTRY = {
    "claude-code": ClaudeCode,
    "qwen-code": QwenCode,
    "qwen-coder": QwenCode,
    "opencode": OpenCode,
}

PROVIDER_API_ENV = {
    "amazon-bedrock": "AWS_ACCESS_KEY_ID",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "github-copilot": "GITHUB_TOKEN",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "HF_TOKEN",
    "llama": "LLAMA_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}


def inject_agent_credentials(
    agent_name: str,
    model_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    extra_env: dict[str, str],
) -> None:
    if agent_name == "claude-code":
        if api_key:
            extra_env.setdefault("ANTHROPIC_API_KEY", api_key)
        if base_url:
            extra_env.setdefault("ANTHROPIC_BASE_URL", base_url)
        return

    if agent_name == "qwen-code":
        if api_key:
            extra_env.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            extra_env.setdefault("OPENAI_BASE_URL", base_url)
        return

    if agent_name == "opencode":
        provider = model_name.split("/", 1)[0] if "/" in model_name else "openai"
        env_key = PROVIDER_API_ENV.get(provider)
        if api_key and env_key:
            extra_env.setdefault(env_key, api_key)
        if base_url and provider == "openai":
            extra_env.setdefault("OPENAI_BASE_URL", base_url)


def build_installed_agent(
    agent_name: str,
    *,
    trial_dir: Path,
    model_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    extra_args: Dict[str, Any],
    reserved_args: set[str],
) -> Any:
    agent_cls = AGENT_REGISTRY[agent_name]
    agent_kwargs = dict(extra_args.get("agent_kwargs") or {})
    for key, value in extra_args.items():
        if key in reserved_args:
            continue
        agent_kwargs.setdefault(key, value)

    extra_env = dict(extra_args.get("agent_env", {}) or {})
    extra_env.update(agent_kwargs.pop("extra_env", {}) or {})
    extra_env.update(agent_kwargs.pop("agent_env", {}) or {})
    inject_agent_credentials(agent_name, model_name, api_key, base_url, extra_env)

    agent_kwargs["logs_dir"] = trial_dir / "agent"
    agent_kwargs["model_name"] = model_name
    agent_kwargs["extra_env"] = extra_env
    return agent_cls(**agent_kwargs)
