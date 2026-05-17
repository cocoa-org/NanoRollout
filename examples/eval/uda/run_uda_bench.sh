#!/usr/bin/env bash
# Run a single migrated benchmark task on the unified uda-desktop image.
#
# The task corpus ships inside the NanoRollout package at
#   nanorollout/envs/uda_env/adapter/<bench>/<instance_id>/
# No external benchmark repo checkout is required at run time.
#
# Switch between benchmarks with BENCH (e.g. BENCH=cocoa-v1, BENCH=osworld-v2
# once that adapter lands).
set -euo pipefail

BENCH="${BENCH:-cocoa-v1}"
INSTANCE_ID="${INSTANCE_ID:-eight-puzzle-game}"
REQUEST_FILE="${REQUEST_FILE:-}"

MODEL_NAME="${MODEL_NAME:-claude-sonnet-4-6}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-}}"
OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-${ANTHROPIC_API_KEY:-}}}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/uda-${BENCH}}"
CONCURRENCY="${CONCURRENCY:-1}"
ENV_TYPE="${ENV_TYPE:-modal}"

UDA_TASKS_DIR="${UDA_TASKS_DIR:-}"
USE_ENCRYPTED_TASKS="${USE_ENCRYPTED_TASKS:-true}"
CLIENT_TYPE="${CLIENT_TYPE:-unified}"

STEP_TIMEOUT="${STEP_TIMEOUT:-600}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
ENV_TIMEOUT="${ENV_TIMEOUT:-180}"
CREATE_TIMEOUT="${CREATE_TIMEOUT:-600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"

cmd=(
  nro run
  --task uda
  --agent uda-agent
  --bench "${BENCH}"
  --model-name "${MODEL_NAME}"
  --env-type "${ENV_TYPE}"
  --output-dir "${OUTPUT_DIR}"
  --concurrency "${CONCURRENCY}"
  --base-url "${OPENAI_BASE_URL}"
  --api-key "${OPENAI_API_KEY}"
  --client-type "${CLIENT_TYPE}"
  --step-timeout "${STEP_TIMEOUT}"
  --eval-timeout "${EVAL_TIMEOUT}"
  --env-timeout "${ENV_TIMEOUT}"
  --create-timeout "${CREATE_TIMEOUT}"
  --max-iterations "${MAX_ITERATIONS}"
  --use-encrypted-tasks "${USE_ENCRYPTED_TASKS}"
)

if [[ -n "${REQUEST_FILE}" ]]; then
  cmd+=(--request-file "${REQUEST_FILE}")
else
  cmd+=(--instance-id "${INSTANCE_ID}")
fi

if [[ -n "${UDA_TASKS_DIR}" ]]; then
  cmd+=(--uda-tasks-dir "${UDA_TASKS_DIR}")
fi

"${cmd[@]}"
