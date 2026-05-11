#!/usr/bin/env bash
set -euo pipefail

# Single-machine NanoRollout HTTP server.
# No `ray start` required: the scheduler runs ray.init(address=auto) on first /run; Ray

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ray start --head

nro serve host=0.0.0.0 port=11000 "$@" 2>&1 | tee logs/server.log
