from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class AgentMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    turns: int = 0
    tool_calls: int = 0
    model_query_time_sum: float = 0.0
    env_execution_time_sum: float = 0.0
    eval_time: float = 0.0
    agent_run_time: float = 0.0
    total_time: float = 0.0


class RunResponse(BaseModel):
    reward: float
    messages: list[dict[str, Any]] = Field(default_factory=list)
    exit_status: str = "unknown"
    agent_metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
