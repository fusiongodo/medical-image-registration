#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

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
    $APT install -y -qq git curl libtiff-dev libjpeg-dev zlib1g-dev rclone
  fi
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip wheel

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "installing CUDA PyTorch..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

echo "installing Python deps (excluding torch stack)..."
grep -vE '^(torch|torchvision|torchaudio)==' setup/requirements.txt > /tmp/requirements_no_torch.txt
pip install -r /tmp/requirements_no_torch.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  ", torch.cuda.get_device_name(0))
PY

pytest introducing_superpoint/tests/test_paths.py -q

echo "setup complete — next: images -> fetch_labels -> quadtree -> keypoint notebooks -> training.py"
