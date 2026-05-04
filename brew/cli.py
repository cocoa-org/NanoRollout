import argparse
import json
import logging
from pathlib import Path
from typing import Any, List
from omegaconf import OmegaConf

import yaml

from brew.core.config import ServerConfig
from brew.core.local import LocalProcessRunner
from brew.core.models import RunRequest
from brew.core.runners import resolve_legacy_runner


def _default_config() -> dict[str, Any]:
    return ServerConfig().model_dump()


def _load_config_file(config_path: str) -> dict[str, Any]:
    if Path(config_path).exists():
        with open(config_path) as handle:
            return yaml.safe_load(handle) or {}
    return {}


def _parse_overrides(overrides: List[str]) -> dict[str, Any]:
    """Parse key=value CLI overrides into a nested config dictionary.

    Supports dot-notation for nested keys, e.g.:
        output_dir=./results/my-exp  port=12000  scheduler.address=auto
    """
    parsed: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        key, raw_value = override.split("=", 1)
        cursor = parsed
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return parsed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(args: argparse.Namespace, overrides: List[str]) -> ServerConfig:
    base_cfg = OmegaConf.create(ServerConfig().model_dump())
    file_cfg = (
        OmegaConf.load(args.config)
        if Path(args.config).exists()
        else OmegaConf.create({})
    )
    cli_cfg = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(base_cfg, file_cfg, cli_cfg)
    resolved = OmegaConf.to_container(merged, resolve=True)
    return ServerConfig(**resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tbrew")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Start the HTTP server")

    run_parser = subparsers.add_parser("run", help="Run instances without HTTP or Ray")
    run_parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Instance id to run. Repeat this flag or pass comma-separated ids.",
    )
    run_parser.add_argument(
        "--request-file",
        default=None,
        help="JSON or JSONL file containing RunRequest objects or request templates.",
    )
    run_parser.add_argument("--model-name", default=None)
    run_parser.add_argument("--task", default="swe")
    run_parser.add_argument("--agent", default="oh-core")
    run_parser.add_argument(
        "--runner",
        default=None,
        help="Deprecated legacy alias. Prefer --task and --agent.",
    )
    run_parser.add_argument("--run-name", default=None)
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the Rich progress UI for local runs.",
    )
    run_parser.add_argument("--base-url", default=None)
    run_parser.add_argument("--api-key", default=None)
    run_parser.add_argument("--env-type", default=None)
    run_parser.add_argument("--task-timeout-s", type=int, default=None)
    run_parser.add_argument("--sampling-params", default=None, help="JSON object")
    run_parser.add_argument("--extra-args", default=None, help="JSON object")
    run_parser.add_argument("--dataset", default="verified")
    run_parser.add_argument("--split", default="test")
    run_parser.add_argument("--step-timeout", type=int, default=600)
    run_parser.add_argument("--eval-timeout", type=int, default=1800)
    run_parser.add_argument("--env-timeout", type=int, default=120)
    run_parser.add_argument("--create-timeout", type=int, default=600)
    run_parser.add_argument("--max-iterations", type=int, default=100)
    return parser


def _load_json_arg(raw: str | None) -> dict | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON argument must be an object")
    return value


def _split_instance_ids(values: list[str]) -> list[str]:
    instance_ids: list[str] = []
    for value in values:
        instance_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return instance_ids


SWE_EXTRA_ARG_FIELDS = {
    "dataset",
    "split",
    "step_timeout",
    "eval_timeout",
    "env_timeout",
    "create_timeout",
    "max_iterations",
}


def _build_extra_args(
    args: argparse.Namespace,
    task: str | None = None,
) -> dict[str, Any]:
    extra_args = _load_json_arg(args.extra_args) or {}
    task = task or _resolve_cli_task(args)
    if task != "swe":
        return extra_args

    defaults = {
        "dataset": args.dataset,
        "split": args.split,
        "step_timeout": args.step_timeout,
        "eval_timeout": args.eval_timeout,
        "env_timeout": args.env_timeout,
        "create_timeout": args.create_timeout,
        "max_iterations": args.max_iterations,
    }
    for key, value in defaults.items():
        extra_args.setdefault(key, value)
    return extra_args


def _resolve_cli_task(args: argparse.Namespace) -> str:
    if args.runner:
        return resolve_legacy_runner(args.runner).task
    return args.task.strip().lower().replace("_", "-")


def _load_request_rows(path: str) -> list[dict[str, Any]]:
    request_path = Path(path).expanduser()
    raw = request_path.read_text(encoding="utf-8")
    if not raw.strip():
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        rows = data if isinstance(data, list) else [data]

    request_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Request file rows must be objects: {request_path}")
        request_rows.append(row)
    return request_rows


def _row_to_request(row: dict[str, Any], args: argparse.Namespace) -> RunRequest:
    row_extra_args = row.get("extra_args") or {}
    if not isinstance(row_extra_args, dict):
        raise ValueError("request-file row extra_args must be an object")

    runner = row.get("runner", args.runner)
    task = row.get("task", args.task)
    agent = row.get("agent", args.agent)
    task_for_defaults = resolve_legacy_runner(runner).task if runner else task
    task_for_defaults = task_for_defaults.strip().lower().replace("_", "-")

    extra_args = _build_extra_args(args, task=task_for_defaults)
    extra_args.update(row_extra_args)
    for key in SWE_EXTRA_ARG_FIELDS:
        if key in row:
            extra_args[key] = row[key]

    model_name = row.get("model_name", args.model_name)
    if not model_name:
        raise ValueError("--model-name is required unless each request row sets model_name")
    if "instance_id" not in row:
        raise ValueError("request-file row is missing instance_id")

    sampling_params = row.get("sampling_params")
    if sampling_params is None:
        sampling_params = _load_json_arg(args.sampling_params)

    request_kwargs = {
        "instance_id": row["instance_id"],
        "model_name": model_name,
        "run_name": row.get("run_name", args.run_name),
        "task_timeout_s": row.get("task_timeout_s", args.task_timeout_s),
        "base_url": row.get("base_url", args.base_url),
        "api_key": row.get("api_key", args.api_key),
        "env_type": row.get("env_type", args.env_type),
        "sampling_params": sampling_params,
        "task": task,
        "agent": agent,
        "runner": runner,
        "extra_args": extra_args,
    }
    if "resources" in row:
        request_kwargs["resources"] = row["resources"]
    return RunRequest(**request_kwargs)


def _build_cli_requests(args: argparse.Namespace) -> list[RunRequest]:
    instance_ids = _split_instance_ids(args.instance_id)
    if not instance_ids:
        return []
    if not args.model_name:
        raise ValueError("--model-name is required when using --instance-id")

    return [
        RunRequest(
            instance_id=instance_id,
            model_name=args.model_name,
            run_name=args.run_name,
            task_timeout_s=args.task_timeout_s,
            base_url=args.base_url,
            api_key=args.api_key,
            env_type=args.env_type,
            sampling_params=_load_json_arg(args.sampling_params),
            task=args.task,
            agent=args.agent,
            runner=args.runner,
            extra_args=_build_extra_args(args),
        )
        for instance_id in instance_ids
    ]


def _build_run_requests(args: argparse.Namespace) -> list[RunRequest]:
    requests: list[RunRequest] = []
    if args.request_file:
        requests.extend(
            _row_to_request(row, args) for row in _load_request_rows(args.request_file)
        )
    requests.extend(_build_cli_requests(args))
    if not requests:
        raise ValueError("Provide --instance-id or --request-file")
    return requests


def _run_direct(config: ServerConfig, args: argparse.Namespace) -> None:
    requests = _build_run_requests(args)
    output_root = args.output_dir or config.output_dir
    runner = LocalProcessRunner(
        output_root=output_root,
        concurrency=args.concurrency,
        show_progress=not args.no_progress,
    )
    responses = runner.run_many(requests)
    payload = [response.model_dump() for response in responses]
    print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))


def _serve(config: ServerConfig) -> None:
    import uvicorn

    from brew.core.server import BrewServer

    server = BrewServer(config)
    app = server.setup_app()
    uvicorn.run(app, host=config.host, port=config.port)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    config = build_config(args, overrides)

    if args.command == "run":
        try:
            _run_direct(config, args)
        except ValueError as exc:
            parser.error(str(exc))
        return

    _serve(config)


if __name__ == "__main__":
    main()
