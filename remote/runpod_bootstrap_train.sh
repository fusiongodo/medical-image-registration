#!/usr/bin/env bash
# Bootstrap a fresh Linux GPU pod: install deps, download + unzip dataset.
# Usage:  bash remote/runpod_bootstrap_train.sh [config_name]
# Then start training with:
#   python introducing_superpoint/training.py <config_name>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-simplepod_v1}"
FILE_ID="13FxvuY782-7pLHm6rl40i41iTDQlINSW"
ZIP_PATH="$REPO_ROOT/data/cropped_smooth_training.zip"
DATA_DIR="$REPO_ROOT/data/cropped_smooth_training"

echo "=== repo:   $REPO_ROOT"
echo "=== config: $CONFIG"

# ── system packages ───────────────────────────────────────────────────────────
if command -v apt-get >/dev/null 2>&1; then
  APT="apt-get"
  [ "$(id -u)" != "0" ] && sudo -n true 2>/dev/null && APT="sudo apt-get"
  $APT update -qq
  $APT install -y -qq python3-venv python3-pip unzip git curl \
      libtiff-dev libjpeg-dev zlib1g-dev
fi

# ── virtualenv ────────────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
  python3 -m venv .venv
  echo "=== venv created"
fi
source .venv/bin/activate
pip install -q --upgrade pip wheel

# ── PyTorch (cu128 required for Blackwell / RTX 5090; works on all ≥ Ampere) ─
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "=== installing PyTorch 2.7 cu128 (~1.1 GB) …"
  pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
fi

# ── project deps ──────────────────────────────────────────────────────────────
echo "=== installing project deps …"
pip install -q -r setup/requirements-linux.txt

# ── dataset ───────────────────────────────────────────────────────────────────
mkdir -p data
if [ ! -f "$ZIP_PATH" ]; then
  echo "=== downloading dataset (4.4 GB) …"
  gdown "$FILE_ID" -O "$ZIP_PATH"
else
  echo "=== zip already present: $ZIP_PATH"
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "=== unzipping …"
  tmp_dir="$(mktemp -d "$REPO_ROOT/data/extract.XXXXXX")"
  unzip -q "$ZIP_PATH" -d "$tmp_dir"

  # zip may contain cropped_smooth_training/ or cropped_smooth/
  extracted="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | grep -v __MACOSX | head -n 1)"
  if [ -z "$extracted" ]; then
    echo "ERROR: no directory found inside zip" >&2; exit 1
  fi
  mv "$extracted" "$DATA_DIR"
  rm -rf "$tmp_dir"
  echo "=== dataset ready: $DATA_DIR"
else
  echo "=== dataset already present: $DATA_DIR"
fi

# ── sanity check ──────────────────────────────────────────────────────────────
python - <<'PY'
from pathlib import Path
import torch

root = Path("data/cropped_smooth_training")
assert root.is_dir(), f"missing {root}"
n = sum(1 for _ in root.glob("*/*/he.png"))
print(f"tiles : {n}")
print(f"torch : {torch.__version__}")
print(f"cuda  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu   : {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "=== setup complete — start training with:"
echo "    python introducing_superpoint/training.py $CONFIG"
