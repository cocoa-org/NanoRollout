#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-trader-joe-2}"
REQUEST_FILE="${REQUEST_FILE:-}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B}"
BASE_URL="${ANTHROPIC_BASE_URL:-${BASE_URL:-}}"
API_KEY="${ANTHROPIC_API_KEY:-${API_KEY:-dummy}}"
AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-dummy}"
AGENT="${AGENT:-claude-code}"
ENV_TYPE="${ENV_TYPE:-modal}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
CONCURRENCY="${CONCURRENCY:-1}"

TASKS_DIR="${TASKS_DIR:-}"
REPO_URL="${REPO_URL:-https://github.com/cocoabench/cocoa-agent.git}"
REPO_DIR="${REPO_DIR:-}"
REPO_REVISION="${REPO_REVISION:-}"
REFRESH_REPO="${REFRESH_REPO:-false}"
TASKS_SUBDIR="${TASKS_SUBDIR:-}"
CLIENT_TYPE="${CLIENT_TYPE:-unified}"
USE_ENCRYPTED_TASKS="${USE_ENCRYPTED_TASKS:-true}"

if [[ -z "${BASE_URL}" ]]; then
  cat >&2 <<'EOF'
BASE_URL or ANTHROPIC_BASE_URL is required for self-hosted runs.

Example:
  BASE_URL="http://127.0.0.1:8000" \
  MODEL_NAME="my-model" \
  bash examples/eval/cocoa/run_claude_code_self_hosted.sh
EOF
  exit 1
fi

cmd=(
  nro run
  --task cocoa-bench
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --agent-env "ANTHROPIC_AUTH_TOKEN=${AUTH_TOKEN}"
  --agent-env "CLAUDE_CODE_OAUTH_TOKEN="
  --agent "${AGENT}"
  --model-name "${MODEL_NAME}"
  --env-type "${ENV_TYPE}"
  --client-type "${CLIENT_TYPE}"
  --use-encrypted-tasks "${USE_ENCRYPTED_TASKS}"
  --repo-url "${REPO_URL}"
  --refresh-repo "${REFRESH_REPO}"
  --output-dir "${OUTPUT_DIR}"
  --concurrency "${CONCURRENCY}"
  --step-timeout "${STEP_TIMEOUT:-1800}"
  --eval-timeout "${EVAL_TIMEOUT:-1800}"
  --env-timeout "${ENV_TIMEOUT:-120}"
  --create-timeout "${CREATE_TIMEOUT:-600}"
  --max-iterations "${MAX_ITERATIONS:-50}"
)

if [[ -n "${THINKING:-}" ]]; then
  if [[ -n "${THINKING_DISPLAY:-}" ]]; then
    cmd+=(--extra-args "{\"thinking\":\"${THINKING}\",\"thinking_display\":\"${THINKING_DISPLAY}\"}")
  else
    cmd+=(--extra-args "{\"thinking\":\"${THINKING}\"}")
  fi
elif [[ -n "${THINKING_DISPLAY:-}" ]]; then
  cmd+=(--extra-args "{\"thinking_display\":\"${THINKING_DISPLAY}\"}")
fi

if [[ -n "${CLAUDE_CODE_ATTRIBUTION_HEADER:-}" ]]; then
  cmd+=(--agent-env "CLAUDE_CODE_ATTRIBUTION_HEADER=${CLAUDE_CODE_ATTRIBUTION_HEADER}")
fi

if [[ -n "${REQUEST_FILE}" ]]; then
  cmd+=(--request-file "${REQUEST_FILE}")
else
  cmd+=(--instance-id "${INSTANCE_ID}")
fi

if [[ -n "${TASKS_DIR}" ]]; then
  cmd+=(--tasks-dir "${TASKS_DIR}")
fi

if [[ -n "${REPO_DIR}" ]]; then
  cmd+=(--repo-dir "${REPO_DIR}")
fi

if [[ -n "${REPO_REVISION}" ]]; then
  cmd+=(--repo-revision "${REPO_REVISION}")
fi

if [[ -n "${TASKS_SUBDIR}" ]]; then
  cmd+=(--tasks-subdir "${TASKS_SUBDIR}")
fi

"${cmd[@]}"
