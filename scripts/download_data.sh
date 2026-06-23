#!/usr/bin/env bash
# Download free LOBSTER sample data for AAPL, AMZN, GOOG, INTC, MSFT.
# After registration at https://lobsterdata.com/info/DataAccess.php you will
# receive a URL. Paste it here or set LOBSTER_SAMPLE_URL in your shell.

set -euo pipefail

DATA_DIR="${LOBSTER_DATA_DIR:-./data/raw}"
mkdir -p "$DATA_DIR"

if [[ -z "${LOBSTER_SAMPLE_URL:-}" ]]; then
  cat <<EOF
LOBSTER_SAMPLE_URL is not set.

Steps:
  1. Register at https://lobsterdata.com/info/DataAccess.php
  2. Copy the download URL you receive by email.
  3. Re-run:  LOBSTER_SAMPLE_URL=<url> bash scripts/download_data.sh

The project will fall back to synthetic data automatically when real files
are absent — you can develop without LOBSTER access.
EOF
  exit 0
fi

echo "Downloading LOBSTER sample into $DATA_DIR"
curl -L -o "$DATA_DIR/lobster_sample.7z" "$LOBSTER_SAMPLE_URL"

if command -v 7z >/dev/null 2>&1; then
  7z x "$DATA_DIR/lobster_sample.7z" -o"$DATA_DIR"
else
  echo "7z not installed — extract $DATA_DIR/lobster_sample.7z manually."
fi
