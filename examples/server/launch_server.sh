#!/usr/bin/env bash
set -euo pipefail

# Single-machine Brew HTTP server.
# No `ray start` required: the scheduler runs ray.init(address=auto) on first /run; Ray
# starts a local runtime automatically unless you point scheduler.address at a cluster.
#
# Usage:
#   bash examples/server/launch_server_single.sh
#   bash examples/server/launch_server_single.sh concurrency=64 port=11000

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

tbrew serve host=0.0.0.0 port=11000 "$@"
