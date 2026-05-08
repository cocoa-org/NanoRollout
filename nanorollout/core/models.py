from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResourceRequest(BaseModel):
    num_cpus: float | None = 1
    num_gpus: float | None = 0
    memory_gb: float | None = 0
    custom: dict[str, float] = Field(default_factory=dict)


class RunRequest(BaseModel):
    instance_id: str
    model_name: str
    run_name: str | None = None
    task_timeout_s: int | None = None
    base_url: str | None = None
    api_key: str | None = None
    env_type: str | None = None
    sampling_params: dict[str, Any] | None = None
    task: str = "swe"
    agent: str = "oh-core"
    runner: str | None = None
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    extra_args: dict[str, Any] | None = None


class RunResponse(BaseModel):
    instance_id: str
    exit_status: str = "unknown"
    output_dir: str | None = None
    error: str | None = None
