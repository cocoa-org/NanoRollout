#!/usr/bin/env bash
# Run a single OSWorld v1 task using the UDA agent on an AWS-backed
# OSWorld VM (EC2 instance booted from the OSWorld-Verified AMI).
#
# Pipeline:
#   UDAAgent (computer_use_* tool calls)
#       -> TaskExecutor (uda env loop)
#       -> OSWorldV1Adapter (pyautogui code translation)
#       -> DesktopEnv (boto3 EC2 lifecycle + osworld-server)
#       -> env.evaluate() => float reward
#
# Tasks ship inside the NanoRollout repo at examples/eval/osworld/data/
# (test_all.json + examples/<domain>/<id>.json), no external checkout.
#
# Prereqs:
#   - AWS creds: AWS_REGION, AWS_SUBNET_ID, AWS_SECURITY_GROUP_ID, AWS_*KEY
#   - boto3 (project dep)
#   - LLM endpoint (Claude/OpenAI/Qwen) for the UDA controller
set -euo pipefail

BENCH="osworld-v1"
INSTANCE_ID="${INSTANCE_ID:-bb5e4c0d-f964-439c-97b6-bdb9747de3f4}"  # chrome: set Bing as default search
REQUEST_FILE="${REQUEST_FILE:-}"

MODEL_NAME="${MODEL_NAME:-claude-sonnet-4-6}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-}}"
OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-${ANTHROPIC_API_KEY:-}}}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/uda-${BENCH}}"
CONCURRENCY="${CONCURRENCY:-1}"
# OSWorld v1 runs against an AWS-backed AMI by default; override with
# ENV_TYPE=docker / azure / gcp / aliyun / volcengine / vmware / virtualbox
# if you have a local OSWorld provider configured.
ENV_TYPE="${ENV_TYPE:-aws}"
REGION="${REGION:-us-east-1}"

# OSWorld v1 corpus is plaintext — no test.py.enc + canary.txt scheme.
USE_ENCRYPTED_TASKS="${USE_ENCRYPTED_TASKS:-false}"

# Route via OSWorldV1Adapter (RuntimeAdapter), not the uda-desktop sandbox.
CLIENT_TYPE="${CLIENT_TYPE:-osworld-v1}"

# Pacing (OSWorld tasks tend to be longer than wildclaw / cocoa).
STEP_TIMEOUT="${STEP_TIMEOUT:-900}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
ENV_TIMEOUT="${ENV_TIMEOUT:-600}"
CREATE_TIMEOUT="${CREATE_TIMEOUT:-900}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"

# Optional viewport overrides (when the model is fine-tuned for a non-default
# resolution). Defaults to 1920x1080 on both sides → CoordScaler is a no-op.
SCREEN_WIDTH="${SCREEN_WIDTH:-1920}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-1080}"
AGENT_VIEW_WIDTH="${AGENT_VIEW_WIDTH:-${SCREEN_WIDTH}}"
AGENT_VIEW_HEIGHT="${AGENT_VIEW_HEIGHT:-${SCREEN_HEIGHT}}"

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
  --extra-args "region=${REGION}"
  --extra-args "screen_width=${SCREEN_WIDTH}"
  --extra-args "screen_height=${SCREEN_HEIGHT}"
  --extra-args "agent_view_width=${AGENT_VIEW_WIDTH}"
  --extra-args "agent_view_height=${AGENT_VIEW_HEIGHT}"
)

if [[ -n "${REQUEST_FILE}" ]]; then
  cmd+=(--request-file "${REQUEST_FILE}")
else
  cmd+=(--instance-id "${INSTANCE_ID}")
fi

"${cmd[@]}"
