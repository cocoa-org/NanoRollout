#!/usr/bin/env bash
set -e

pip install openai playwright
python3 -m playwright install chromium
