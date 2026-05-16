#!/usr/bin/env bash
set -e

pip install -q fastapi uvicorn 2>/dev/null
export SLACK_FIXTURES=/tmp_workspace/tmp/messages.json && python3 /tmp_workspace/mock_services/slack/server.py &
sleep 2
rm -f -r /tmp_workspace/tmp
