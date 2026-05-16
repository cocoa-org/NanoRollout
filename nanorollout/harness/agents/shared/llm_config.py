"""Shared LLM config helpers for agent construction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    llm_provider: str = "openai"
    temperature: float = 0.6
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_body: dict[str, Any] = field(default_factory=dict)


def parse_json_object(value: Optional[object]) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def build_llm_config(
    *,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    sampling_params: Optional[object],
    llm_provider: str = "openai",
    default_temperature: float = 0.6,
    default_top_p: Optional[float] = None,
    default_max_tokens: Optional[int] = None,
) -> LLMConfig:
    params = parse_json_object(sampling_params)
    return LLMConfig(
        model=model,
        api_key=api_key,
        api_base=base_url,
        llm_provider=params.get("llm_provider", llm_provider),
        temperature=params.get("temperature", default_temperature),
        top_p=params.get("top_p", default_top_p),
        max_tokens=params.get("max_tokens", default_max_tokens),
        extra_body=params.get("extra_body", {}),
    )
