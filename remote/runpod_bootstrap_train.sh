#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FILE_ID="13FxvuY782-7pLHm6rl40i41iTDQlINSW"
ZIP_PATH="$REPO_ROOT/data/cropped_smooth_training.zip"
CROPPED_DIR="$REPO_ROOT/data/cropped_smooth"

echo "repo: $REPO_ROOT"

if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    APT="apt-get"
  elif sudo -n true 2>/dev/null; then
    APT="sudo apt-get"
  else
    APT=""
  fi
  if [ -n "$APT" ]; then
    $APT update -qq
    $APT install -y -qq unzip git curl libtiff-dev libjpeg-dev zlib1g-dev
  fi
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install -U pip wheel
python -m pip install -r setup/requirements-freeze.txt
python -m pip install gdown

mkdir -p data
if [ ! -f "$ZIP_PATH" ]; then
  gdown --id "$FILE_ID" -O "$ZIP_PATH"
else
  echo "archive already exists: $ZIP_PATH"
fi

if [ ! -d "$CROPPED_DIR" ]; then
  tmp_dir="$(mktemp -d "$REPO_ROOT/data/cropped_smooth_extract.XXXXXX")"
  unzip -q "$ZIP_PATH" -d "$tmp_dir"

  if [ -d "$tmp_dir/cropped_smooth" ]; then
    mv "$tmp_dir/cropped_smooth" "$CROPPED_DIR"
  elif [ -d "$tmp_dir/cropped_smooth_training" ]; then
    mv "$tmp_dir/cropped_smooth_training" "$CROPPED_DIR"
  else
    first_dir="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -z "$first_dir" ]; then
      echo "could not find extracted cropped_smooth directory" >&2
      exit 1
    fi
    mv "$first_dir" "$CROPPED_DIR"
  fi
  rm -rf "$tmp_dir"
else
  echo "data already exists: $CROPPED_DIR"
fi

test -d "$CROPPED_DIR"

python - <<'PY'
from pathlib import Path
import torch

root = Path("data/cropped_smooth")
assert root.is_dir(), f"missing {root}"
print("tiles:", sum(1 for _ in root.glob("*/*/*/he.png")))
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
PY

cd "$REPO_ROOT/introducing_superpoint"
exec python training.py
