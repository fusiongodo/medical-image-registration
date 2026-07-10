#!/usr/bin/env bash
# Runs the find_w_fp config batch sequentially, unattended. Each training.py
# invocation trains for its own num_epochs and exits on its own; this script
# just chains them and keeps going even if one run fails, so a bad run
# doesn't block the rest of the batch overnight.
cd "$(dirname "$0")/.."
source .venv/bin/activate

configs=(
  find_w_fp/fp05
  find_w_fp/fp1
  find_w_fp/fp2
  find_w_fp/fp4
)

for cfg in "${configs[@]}"; do
  name=$(basename "$cfg")
  echo "=== $(date) starting $cfg ==="
  python -u introducing_superpoint/training.py "$cfg" \
    --monitor-log "${name}_monitor.log" \
    >> "${name}_stdout.log" 2>&1
  echo "=== $(date) finished $cfg (exit $?) ==="
done

echo "=== $(date) sweep complete ==="
