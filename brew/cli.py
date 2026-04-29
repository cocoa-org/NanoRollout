import argparse
import logging
from pathlib import Path
from typing import List

import uvicorn
from omegaconf import DictConfig, OmegaConf

from tinyflow.core.config import ServerConfig
from tinyflow.runtime.server import TinyFlowServer


def _default_config() -> DictConfig:
    return OmegaConf.create(ServerConfig().model_dump())


def _load_config_file(config_path: str) -> DictConfig:
    if Path(config_path).exists():
        return OmegaConf.load(config_path)
    return OmegaConf.create({})


def _parse_overrides(overrides: List[str]) -> DictConfig:
    """Parse key=value CLI overrides into an OmegaConf DictConfig.

    Supports dot-notation for nested keys, e.g.:
        output_dir=./results/my-exp  port=12000  monitoring.enabled=true
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
    parser = argparse.ArgumentParser(description="TinyFlow")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    config = build_config(args, overrides)

    server = TinyFlowServer(config)
    app = server.setup_app()

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
