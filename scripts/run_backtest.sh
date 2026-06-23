#!/usr/bin/env bash
# Run backtest with the latest checkpoint.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/best_model.pt}"
CONFIG="${CONFIG:-configs/tcn_small.yaml}"
HORIZON="${HORIZON:-10}"

python scripts/run_backtest.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --horizon "$HORIZON"
