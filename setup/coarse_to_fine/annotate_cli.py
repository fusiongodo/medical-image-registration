"""
Persist a single human refinement action for the coarse-to-fine UI.

Appends (or clears) one durable annotation entry in
data/registration_annotations.json.  A tile carries at most one entry per level,
so approve/correct/exclude replace any prior entry for the same (level, tile_loc).

Usage:
    python setup/coarse_to_fine/annotate_cli.py <pair> <level> <tile_loc> approve <u> <v>
    python setup/coarse_to_fine/annotate_cli.py <pair> <level> <tile_loc> correct <u> <v>
    python setup/coarse_to_fine/annotate_cli.py <pair> <level> <tile_loc> exclude [<u> <v>]
    python setup/coarse_to_fine/annotate_cli.py <pair> <level> <tile_loc> clear

`approve` stores the FFT candidate displacement (source=fft), `correct` stores a
human landmark-derived displacement (source=human), `exclude` marks the tile as
deliberately ignored (source=human, conf=0.0).  Displacements <u>, <v> are
tile-pixel dx, dy.  Prints {"ok": true, "count": <entries for pair>}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf  # noqa: E402,F401  (path bootstrap for the package import below)

from setup.coarse_to_fine import annotations


def annotate(pair_id: int, level: int, tile_loc: str, action: str, u: float, v: float) -> dict:
    annotations.remove(pair_id, level, tile_loc)
    if action == "clear":
        return {"ok": True, "count": len(annotations.load(pair_id))}

    if action == "exclude":
        entry = annotations.make_entry(level, tile_loc, u, v, action, "human", 0.0)
    elif action == "correct":
        entry = annotations.make_entry(level, tile_loc, u, v, action, "human", 1.0)
    else:  # approve
        entry = annotations.make_entry(level, tile_loc, u, v, action, "fft", 1.0)
    entries = annotations.add(pair_id, entry)
    return {"ok": True, "count": len(entries)}


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 4:
        sys.exit("Usage: annotate_cli.py <pair> <level> <tile_loc> approve|correct <u> <v> | clear")

    pair_id, level, tile_loc, action = int(argv[0]), int(argv[1]), argv[2], argv[3]
    if action not in ("approve", "correct", "exclude", "clear"):
        sys.exit(f"unknown action: {action}")

    u, v = 0.0, 0.0
    if action not in ("clear", "exclude"):
        if len(argv) < 6:
            sys.exit(f"{action} requires <u> <v>")
        u, v = float(argv[4]), float(argv[5])
    elif action == "exclude" and len(argv) >= 6:
        u, v = float(argv[4]), float(argv[5])

    print(json.dumps(annotate(pair_id, level, tile_loc, action, u, v), separators=(",", ":")))


if __name__ == "__main__":
    main()
