"""CLI wrapper: eval TRE matrix for pair + batch. Prints JSON to stdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from setup.coarse_to_fine.eval_tre import compute_batch_pair_tre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", type=int)
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    print(json.dumps(compute_batch_pair_tre(args.batch, args.pair), separators=(",", ":")))


if __name__ == "__main__":
    main()
