from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class RayRuntimeConfig(BaseModel):
    address: str = "auto"
    namespace: Optional[str] = None
    runtime_env: Optional[Dict[str, Any]] = None


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "tinyflow-monitor"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "cluster-monitor"
    tags: list[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    enabled: bool = False
    interval_s: float = 30.0
    print_summary: bool = True
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 11000
    concurrency: int = 256
    max_pending_jobs: int = 0

    env: str = "enroot"
    step_timeout: int = 600
    eval_timeout: int = 1800
    step_limit: int = 100

    output_dir: str = "./results"
    skip_if_exists: bool = False

    ray: RayRuntimeConfig = Field(default_factory=RayRuntimeConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
