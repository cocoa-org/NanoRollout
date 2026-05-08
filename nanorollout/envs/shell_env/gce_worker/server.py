"""GCE worker daemon entry point.

Run on each MIG worker VM via systemd ExecStart::

    /usr/bin/python3 -m brew.envs.shell_env.gce_worker.server --port 50051

Listens on 0.0.0.0:50051 with insecure credentials (VPC-internal only)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

import grpc

from .proto import worker_pb2_grpc
from .service import WORKER_VERSION, WorkerService

DEFAULT_PORT = 50051


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GCE worker daemon")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="Address to bind. Use 127.0.0.1 for tests, 0.0.0.0 in MIG.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


async def _serve(bind: str, port: int) -> None:
    server = grpc.aio.server()
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerService(), server)
    listen_addr = f"{bind}:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logging.info("worker daemon v%s listening on %s", WORKER_VERSION, listen_addr)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        await stop_event.wait()
    finally:
        logging.info("shutdown signal received, draining...")
        await server.stop(grace=5)
        logging.info("worker daemon stopped")


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_serve(args.bind, args.port))


if __name__ == "__main__":
    main()
