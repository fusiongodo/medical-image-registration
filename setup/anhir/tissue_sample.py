"""Deterministic ≥1/3-per-tissue ANHIR training sample."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def tissue_of(case: str) -> str:
    m = re.match(r"(.+)_(\d+)$", case or "")
    return m.group(1) if m else (case or "")


def sample_third(pairs: list[dict]) -> tuple[list[int], dict]:
    by: dict[str, list[int]] = defaultdict(list)
    for p in pairs:
        by[tissue_of(str(p.get("case") or ""))].append(int(p["id"]))
    breakdown: dict = {}
    chosen: list[int] = []
    for tissue, ids in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n = len(ids)
        k = max(1, round(n / 3))
        take = ids[:k]
        breakdown[tissue] = {"n": n, "k": k, "ids": take}
        chosen.extend(take)
    return sorted(chosen), breakdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "anhir_tissue77.json",
    )
    args = ap.parse_args()
    from setup import datasets

    pairs = datasets.load_pairs("anhir")
    ids, breakdown = sample_third(pairs)
    payload = {
        "dataset": "anhir",
        "n": len(ids),
        "pairs": ids,
        "by_tissue": breakdown,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"n": len(ids), "pairs": ids, "out": str(args.out)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
