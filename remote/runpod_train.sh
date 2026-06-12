#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/introducing_superpoint"
source "$REPO_ROOT/.venv/bin/activate"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — pick a GPU pod"
print("gpu:", torch.cuda.get_device_name(0))
PY

exec python training.py
