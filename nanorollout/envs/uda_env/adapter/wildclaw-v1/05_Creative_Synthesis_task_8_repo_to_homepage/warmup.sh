#!/usr/bin/env bash
set -e

pip install "requests" "beautifulsoup4" "Pillow" "playwright"
playwright install chromium
