import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List

import uvicorn
from omegaconf import DictConfig, OmegaConf

from brew.core.config import ServerConfig
from brew.core.models import RunRequest
from brew.core.server import BrewServer


def _default_config() -> DictConfig:
    return OmegaConf.create(ServerConfig().model_dump())


def _load_config_file(config_path: str) -> DictConfig:
    if Path(config_path).exists():
        return OmegaConf.load(config_path)
    return OmegaConf.create({})


def _parse_overrides(overrides: List[str]) -> DictConfig:
    """Parse key=value CLI overrides into an OmegaConf DictConfig.

    Supports dot-notation for nested keys, e.g.:
        output_dir=./results/my-exp  port=12000  scheduler.address=auto
    """
    return OmegaConf.from_dotlist(overrides)


def build_config(args: argparse.Namespace, overrides: List[str]) -> ServerConfig:
    base_cfg = _default_config()
    file_cfg = _load_config_file(args.config)
    cli_cfg = _parse_overrides(overrides)
    merged = OmegaConf.merge(base_cfg, file_cfg, cli_cfg)
    resolved = OmegaConf.to_container(merged, resolve=True)
    return ServerConfig(**resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brew")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Start the HTTP server")

    run_parser = subparsers.add_parser("run", help="Run one instance without HTTP")
    run_parser.add_argument("--instance-id", required=True)
    run_parser.add_argument("--model-name", required=True)
    run_parser.add_argument("--runner", default="oh-core")
    run_parser.add_argument("--run-name", default=None)
    run_parser.add_argument("--base-url", default=None)
    run_parser.add_argument("--api-key", default=None)
    run_parser.add_argument("--env-type", default=None)
    run_parser.add_argument("--task-timeout-s", type=int, default=None)
    run_parser.add_argument("--sampling-params", default=None, help="JSON object")
    run_parser.add_argument("--extra-args", default=None, help="JSON object")
    return parser


def _load_json_arg(raw: str | None) -> dict | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON argument must be an object")
    return value


async def _run_once(config: ServerConfig, args: argparse.Namespace) -> None:
    server = BrewServer(config)
    request = RunRequest(
        instance_id=args.instance_id,
        model_name=args.model_name,
        run_name=args.run_name,
        task_timeout_s=args.task_timeout_s,
        base_url=args.base_url,
        api_key=args.api_key,
        env_type=args.env_type,
        sampling_params=_load_json_arg(args.sampling_params),
        runner=args.runner,
        extra_args=_load_json_arg(args.extra_args),
    )
    try:
        response = await server.handle_run(request)
    finally:
        server.close()
    print(json.dumps(response.model_dump(), indent=2))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    config = build_config(args, overrides)

    if args.command == "run":
        asyncio.run(_run_once(config, args))
        return

    server = BrewServer(config)
    app = server.setup_app()
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
