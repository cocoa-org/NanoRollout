#!/usr/bin/env bash
set -e

pip install -q fastapi uvicorn 2>/dev/null
export GMAIL_FIXTURES=/tmp_workspace/fixtures/gmail/inbox.json && python3 /tmp_workspace/mock_services/gmail/server.py &
export CALENDAR_FIXTURES=/tmp_workspace/fixtures/calendar/events.json && python3 /tmp_workspace/mock_services/calendar/server.py &
sleep 3
rm -rf /tmp_workspace/fixtures
