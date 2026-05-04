from typing import Any

from pydantic import BaseModel, Field


class SchedulerConfig(BaseModel):
    address: str = "auto"
    namespace: str | None = None
    runtime_env: dict[str, Any] | None = None


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 11000
    concurrency: int = 256
    output_dir: str = "./results"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
