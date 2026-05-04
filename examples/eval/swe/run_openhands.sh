#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-django__django-11095}"
REQUEST_FILE="${REQUEST_FILE:-}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-https://api.deepseek.com/v1}}"
OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-abc-123}}"
AGENT="${AGENT:-OpenHands}"
DATASET="${DATASET:-verified}"
ENV_TYPE="${ENV_TYPE:-modal}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
CONCURRENCY="${CONCURRENCY:-1}"

cmd=(
  tbrew run
  --task swe
  --agent "${AGENT}"
  --model-name "${MODEL_NAME}"
  --base-url "${OPENAI_BASE_URL}"
  --api-key "${OPENAI_API_KEY}"
  --env-type "${ENV_TYPE}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --output-dir "${OUTPUT_DIR}"
  --concurrency "${CONCURRENCY}"
  --step-timeout "${STEP_TIMEOUT:-600}"
  --eval-timeout "${EVAL_TIMEOUT:-1800}"
  --env-timeout "${ENV_TIMEOUT:-120}"
  --create-timeout "${CREATE_TIMEOUT:-600}"
  --max-iterations "${MAX_ITERATIONS:-100}"
)

if [[ -n "${REQUEST_FILE}" ]]; then
  cmd+=(--request-file "${REQUEST_FILE}")
else
  cmd+=(--instance-id "${INSTANCE_ID}")
fi

"${cmd[@]}"
