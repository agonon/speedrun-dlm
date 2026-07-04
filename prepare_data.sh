#!/usr/bin/env bash
set -euo pipefail

TARGET_TOKENS=${1:-393216000}
CHUNK_TOKENS=100000000
TRAIN_CHUNKS=$(( (TARGET_TOKENS + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))

python data/cached_fineweb10B.py "$TRAIN_CHUNKS"
