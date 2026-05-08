#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-adaptive-rejection-sampler}"
REQUEST_FILE="${REQUEST_FILE:-}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-}}"
OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"
AGENT="${AGENT:-terminus2}"
ENV_TYPE="${ENV_TYPE:-modal}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
CONCURRENCY="${CONCURRENCY:-1}"

REPO_URL="${REPO_URL:-https://github.com/harbor-framework/terminal-bench-2.git}"
REPO_DIR="${REPO_DIR:-}"
REPO_REVISION="${REPO_REVISION:-}"
REFRESH_REPO="${REFRESH_REPO:-false}"

STEP_TIMEOUT="${STEP_TIMEOUT:--1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:--1}"
ENV_TIMEOUT="${ENV_TIMEOUT:-120}"
CREATE_TIMEOUT="${CREATE_TIMEOUT:-600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
TIMEOUT_MULTIPLIER="${TIMEOUT_MULTIPLIER:-1.0}"
PARSER_NAME="${PARSER_NAME:-json}"

cmd=(
  nro run
  --task terminal
  --agent "${AGENT}"
  --model-name "${MODEL_NAME}"
  --env-type "${ENV_TYPE}"
  --output-dir "${OUTPUT_DIR}"
  --concurrency "${CONCURRENCY}"
  --base-url "${OPENAI_BASE_URL}"
  --api-key "${OPENAI_API_KEY}"
  --instance-id "${INSTANCE_ID}"
  --request-file "${REQUEST_FILE}"
  --step-timeout "${STEP_TIMEOUT}"
  --eval-timeout "${EVAL_TIMEOUT}"
  --env-timeout "${ENV_TIMEOUT}"
  --create-timeout "${CREATE_TIMEOUT}"
  --max-iterations "${MAX_ITERATIONS}"
  --repo-url "${REPO_URL}"
  --repo-dir "${REPO_DIR}"
  --repo-revision "${REPO_REVISION}"
  --refresh-repo "${REFRESH_REPO}"
  --parser-name "${PARSER_NAME}"
  --timeout-multiplier "${TIMEOUT_MULTIPLIER}"
)

"${cmd[@]}"
