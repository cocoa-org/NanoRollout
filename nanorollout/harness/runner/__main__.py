"""CLI entry point for NanoRollout runner.

Usage: python -m nanorollout.harness.runner [OPTIONS]
"""

import argparse
import json
import logging
import os
from pathlib import Path

from nanorollout.harness.runner.swe.oh_core import run_oh_core


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single OH-Core instance via NanoRollout runner."
    )
    parser.add_argument("--instance-id", default="django__django-11095", help="Target instance id.")
    parser.add_argument("--output-dir", default="runs/debug", help="Output directory.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Model name.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_API_BASE", "http://10.24.3.144:8000/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "abc-123"),
        help="API key (optional for local servers).",
    )
    parser.add_argument(
        "--env-type",
        default="enroot",
        choices=["docker", "enroot", "singularity", "modal"],
        help="Execution environment type.",
    )
    parser.add_argument("--dataset", default="verified", help="Dataset subset name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--step-timeout", type=int, default=600, help="Step timeout.")
    parser.add_argument("--eval-timeout", type=int, default=1800, help="Eval timeout.")
    parser.add_argument(
        "--env-timeout",
        type=int,
        default=120,
        help="Environment command timeout.",
    )
    parser.add_argument(
        "--create-timeout",
        type=int,
        default=600,
        help="Container create timeout.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100, help="Max agent iterations."
    )
    parser.add_argument(
        "--sampling-params",
        default=None,
        help="JSON string of sampling parameters.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_oh_core(
        instance_id=args.instance_id,
        output_dir=str(output_dir),
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        env_type=args.env_type,
        sampling_params=args.sampling_params,
        extra_args={
            "step_timeout": args.step_timeout,
            "eval_timeout": args.eval_timeout,
            "env_timeout": args.env_timeout,
            "create_timeout": args.create_timeout,
            "max_iterations": args.max_iterations,
            "dataset": args.dataset,
            "split": args.split,
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
