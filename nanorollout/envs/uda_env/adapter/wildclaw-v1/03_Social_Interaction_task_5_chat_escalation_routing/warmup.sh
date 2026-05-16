#!/usr/bin/env bash
set -e

pip install -q fastapi uvicorn
export SLACK_FIXTURES=/tmp_workspace/tmp/messages.json && python3 /tmp_workspace/mock_services/slack/server.py &
export CONTACTS_FIXTURES=/tmp_workspace/tmp/contacts.json && python3 /tmp_workspace/mock_services/contacts/server.py &
sleep 3
rm -f -r /tmp_workspace/tmp
