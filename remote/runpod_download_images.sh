#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found. Run remote/runpod_setup.sh first."
  exit 1
fi

source .venv/bin/activate

python remote/gdrive_sync.py setup
python remote/gdrive_sync.py download "$REPO_ROOT/data" --images-only

echo "images in $REPO_ROOT/data/images"
ls -lh "$REPO_ROOT/data/images/"*.data 2>/dev/null | wc -l | xargs echo "file count:"
