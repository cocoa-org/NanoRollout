from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResourceRequest(BaseModel):
    num_cpus: Optional[float] = 1
    num_gpus: Optional[float] = 0
    memory_gb: Optional[float] = 0
    resources: Dict[str, float] = Field(default_factory=dict)


class RunRequest(BaseModel):
    instance_id: str
    model_name: str
    run_name: Optional[str] = None
    task_timeout_s: Optional[int] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    env_type: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    runner: Optional[str] = None
    runner_module: Optional[str] = None
    runner_entrypoint: Optional[str] = None
    task_type: Optional[str] = None
    runtime_env: Optional[Dict[str, Any]] = None
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    # extra args for the runner
    extra_args: Optional[Dict[str, Any]] = None


class AgentMetrics(BaseModel):
    turns: int = 0
    tool_calls: int = 0
    model_query_time_sum: float = 0.0
    env_execution_time_sum: float = 0.0
    eval_time: float = 0.0
    agent_run_time: float = 0.0
    total_time: float = 0.0


class RunResponse(BaseModel):
    reward: float
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    exit_status: str = "unknown"
    agent_metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tools: Optional[List[Dict[str, Any]]] = None
