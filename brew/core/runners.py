import importlib
import json
from dataclasses import dataclass
from typing import Any

from brew.core.models import RunRequest


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


@dataclass(frozen=True)
class RunnerSpec:
    task: str
    agent: str
    module: str
    entrypoint: str
    aliases: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize_name(self.task), _normalize_name(self.agent))


RUNNER_SPECS: tuple[RunnerSpec, ...] = (
    RunnerSpec(
        task="swe",
        agent="OpenHands",
        module="brew.harness.runner.swe.oh_core",
        entrypoint="run_oh_core",
        aliases=("openhands", "oh-core",),
    ),
    RunnerSpec(
        task="swe",
        agent="OpenHands-Lite",
        module="brew.harness.runner.swe.oh_lite",
        entrypoint="run_oh_lite",
        aliases=("openhands-lite", "oh-lite"),
    ),
    RunnerSpec(
        task="swe",
        agent="mini-swe-agent",
        module="brew.harness.runner.swe.miniswe",
        entrypoint="run_miniswe",
        aliases=("miniswe", "minisweagent"),
    ),
    RunnerSpec(
        task="swe",
        agent="r2egym",
        module="brew.harness.runner.swe.r2egym",
        entrypoint="run_r2egym",
        aliases=("r2e-gym",),
    ),
    RunnerSpec(
        task="osworld",
        agent="qwen3vl-mmagents",
        module="brew.harness.runner.osworld",
        entrypoint="run_osworld",
        # aliases=("osworld"),
    ),
)


RUNNER_REGISTRY: dict[tuple[str, str], RunnerSpec] = {}
for spec in RUNNER_SPECS:
    RUNNER_REGISTRY[spec.key] = spec
    for alias in spec.aliases:
        RUNNER_REGISTRY[(_normalize_name(spec.task), _normalize_name(alias))] = spec

LEGACY_RUNNER_ALIASES: dict[str, tuple[str, str]] = {}
for spec in RUNNER_SPECS:
    LEGACY_RUNNER_ALIASES[_normalize_name(spec.agent)] = spec.key
    for alias in spec.aliases:
        LEGACY_RUNNER_ALIASES[_normalize_name(alias)] = spec.key

def resolve_runner(task: str, agent: str) -> RunnerSpec:
    key = (_normalize_name(task), _normalize_name(agent))
    try:
        return RUNNER_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported task/agent pair: {task}/{agent}") from exc


def resolve_legacy_runner(runner: str) -> RunnerSpec:
    runner_name = _normalize_name(runner)
    try:
        key = LEGACY_RUNNER_ALIASES[runner_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported runner type: {runner_name}") from exc
    return RUNNER_REGISTRY[key]


def resolve_request_runner(request: RunRequest) -> RunnerSpec:
    if request.runner:
        return resolve_legacy_runner(request.runner)
    return resolve_runner(request.task, request.agent)


def load_runner_callable(module_name: str, entrypoint: str):
    module = importlib.import_module(module_name)
    try:
        return getattr(module, entrypoint)
    except AttributeError as exc:
        raise ImportError(
            f"Runner entrypoint {entrypoint!r} not found in module {module_name!r}"
        ) from exc


def build_runner_params(request: RunRequest, output_dir: str) -> dict[str, Any]:
    spec = resolve_request_runner(request)
    extra_args = dict(request.extra_args or {})
    if spec.task == "osworld":
        extra_args.setdefault("agent", spec.agent)

    params: dict[str, Any] = {
        "instance_id": request.instance_id,
        "model_name": request.model_name,
        "sampling_params": json.dumps(request.sampling_params or {}),
        "output_dir": output_dir,
        "extra_args": extra_args,
    }
    if request.base_url is not None:
        params["base_url"] = request.base_url
    if request.api_key is not None:
        params["api_key"] = request.api_key
    if request.env_type is not None:
        params["env_type"] = request.env_type
    return params
