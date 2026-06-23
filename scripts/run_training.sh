#!/usr/bin/env bash
# Build a synthetic dataset and train the TCN end-to-end.
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
EVENTS="${EVENTS:-200000}"

python scripts/build_dataset.py --synthetic --n-events "$EVENTS"
python scripts/train.py --config "$CONFIG"
