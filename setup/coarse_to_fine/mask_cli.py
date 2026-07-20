"""
Persist / query the "mask out" state for the coarse-to-fine UI.

Masks propagate forward to descendant tiles by quadtree index (see masks.py).

Usage:
    python setup/coarse_to_fine/mask_cli.py <pair> <level> <tile_loc> mask|unmask|clear
    python setup/coarse_to_fine/mask_cli.py <pair> <level> list

The write actions print {"ok": true, "masked": <bool>, "count": <entries>}.
`list` prints the effective masked map for the full grid at that level:
{"x_y": true, ...} (only masked tiles are included).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf  # noqa: E402,F401  (path bootstrap for the package import below)

from setup.coarse_to_fine import masks


def _list(pair_id: int, level: int) -> dict:
    entries = masks.load(pair_id)
    grid = 2 ** level
    out: dict[str, bool] = {}
    for y in range(grid):
        for x in range(grid):
            loc = f"{x}_{y}"
            if masks.is_masked(entries, level, loc):
                out[loc] = True
    return out


def _write(pair_id: int, level: int, tile_loc: str, action: str) -> dict:
    if action == "clear":
        masked = masks.clear(pair_id, level, tile_loc)
    else:
        masked = masks.set_effective(pair_id, level, tile_loc, action == "mask")
    return {"ok": True, "masked": masked, "count": len(masks.load(pair_id))}


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 3:
        sys.exit("Usage: mask_cli.py <pair> <level> <tile_loc> mask|unmask|clear | <pair> <level> list")

    pair_id, level = int(argv[0]), int(argv[1])

    if argv[2] == "list":
        result = _list(pair_id, level)
    else:
        if len(argv) < 4:
            sys.exit("Usage: mask_cli.py <pair> <level> <tile_loc> mask|unmask|clear")
        tile_loc, action = argv[2], argv[3]
        if action not in ("mask", "unmask", "clear"):
            sys.exit(f"unknown action: {action}")
        result = _write(pair_id, level, tile_loc, action)

    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
