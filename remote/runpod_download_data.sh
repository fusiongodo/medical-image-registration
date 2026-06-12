#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export DATA_PREFIX="${DATA_PREFIX:-macos}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found. Run remote/runpod_setup.sh first or: curl https://rclone.org/install.sh | sudo bash"
  exit 1
fi

source .venv/bin/activate 2>/dev/null || true

python remote/gdrive_sync.py setup
python remote/gdrive_sync.py download "$REPO_ROOT/data"

echo "data ready under $REPO_ROOT/data"
echo "annotations: data/${DATA_PREFIX}_*.json"
ls -lh "$REPO_ROOT/data/"*.json 2>/dev/null || true
