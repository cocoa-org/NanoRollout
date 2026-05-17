import importlib
import json
from dataclasses import dataclass
from typing import Any

from nanorollout.core.models import RunRequest

SWE_RUNNER_MODULE = "nanorollout.adapters.swe.entrypoints"
TERMINAL_RUNNER_MODULE = "nanorollout.adapters.terminal.entrypoints"


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


def _swe_runner(
    agent: str,
    entrypoint: str,
    aliases: tuple[str, ...] = (),
) -> RunnerSpec:
    return RunnerSpec(
        task="swe",
        agent=agent,
        module=SWE_RUNNER_MODULE,
        entrypoint=entrypoint,
        aliases=aliases,
    )


def _terminal_runner(
    agent: str,
    entrypoint: str,
    aliases: tuple[str, ...] = (),
) -> RunnerSpec:
    return RunnerSpec(
        task="terminal",
        agent=agent,
        module=TERMINAL_RUNNER_MODULE,
        entrypoint=entrypoint,
        aliases=aliases,
    )


RUNNER_SPECS: tuple[RunnerSpec, ...] = (
    _swe_runner(
        "OpenHands",
        "run_oh_core",
        aliases=(
            "openhands",
            "oh-core",
        ),
    ),
    _swe_runner(
        "OpenHands-Lite",
        "run_oh_lite",
        aliases=("openhands-lite", "oh-lite"),
    ),
    _swe_runner(
        "mini-swe-agent",
        "run_miniswe",
        aliases=("miniswe", "minisweagent"),
    ),
    _swe_runner(
        "claude-code",
        "run_installed_claude_code",
        aliases=("claudecode", "swe-claude-code"),
    ),
    _swe_runner(
        "qwen-code",
        "run_installed_qwen_code",
        aliases=("qwen-coder", "swe-qwen-code"),
    ),
    _swe_runner(
        "opencode",
        "run_installed_opencode",
        aliases=("open-code", "swe-opencode"),
    ),
    _swe_runner(
        "r2egym",
        "run_r2egym",
        aliases=("r2e-gym",),
    ),
    RunnerSpec(
        task="osworld",
        agent="qwen3vl-mmagents",
        module="nanorollout.adapters.osworld.entrypoints",
        entrypoint="run_osworld",
        # aliases=("osworld"),
    ),
    _terminal_runner(
        "mini-swe-agent",
        "run_tb_miniswe",
        aliases=("miniswe", "terminal-miniswe"),
    ),
    _terminal_runner(
        "claude-code",
        "run_tb_claude_code",
        aliases=("claudecode", "terminal-claude-code"),
    ),
    _terminal_runner(
        "qwen-code",
        "run_tb_qwen_code",
        aliases=("qwen-coder", "terminal-qwen-code"),
    ),
    _terminal_runner(
        "opencode",
        "run_tb_opencode",
        aliases=("open-code", "terminal-opencode"),
    ),
    _terminal_runner(
        "terminus2",
        "run_tb_terminus2",
        aliases=("terminus-2",),
    ),
    RunnerSpec(
        task="cocoa-bench",
        agent="cocoa-agent",
        module="nanorollout.adapters.cocoa.entrypoints",
        entrypoint="run_cocoa_agent",
        aliases=("cocoa", "cocoagent"),
    ),
    RunnerSpec(
        task="uda",
        agent="uda-agent",
        module="nanorollout.harness.runner.uda.uda_agent",
        entrypoint="run_uda_agent",
        aliases=("uda", "udaagent", "uda_agent"),
    ),
)


RUNNER_REGISTRY: dict[tuple[str, str], RunnerSpec] = {}
for spec in RUNNER_SPECS:
    RUNNER_REGISTRY[spec.key] = spec
    for alias in spec.aliases:
        RUNNER_REGISTRY[(_normalize_name(spec.task), _normalize_name(alias))] = spec

LEGACY_RUNNER_ALIASES: dict[str, tuple[str, str]] = {}
for spec in RUNNER_SPECS:
    LEGACY_RUNNER_ALIASES.setdefault(_normalize_name(spec.agent), spec.key)
    for alias in spec.aliases:
        LEGACY_RUNNER_ALIASES.setdefault(_normalize_name(alias), spec.key)


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
