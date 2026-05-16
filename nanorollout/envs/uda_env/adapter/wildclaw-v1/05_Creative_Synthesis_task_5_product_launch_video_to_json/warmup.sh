#!/usr/bin/env bash
set -e

pip install "weasyprint" "pymupdf" "requests"
apt-get update && apt-get install -y ffmpeg
