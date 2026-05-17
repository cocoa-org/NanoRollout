#!/usr/bin/env bash
set -e

pip install "openai>=1.0"
apt-get update && apt-get install -y ffmpeg
