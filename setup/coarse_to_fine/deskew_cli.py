"""
CLI for the global rotation-free deskew (see deskew.py).

All commands print a single JSON result line to stdout.

Usage:
    # apply: correspondence points arrive as JSON on stdin, either a bare list
    #   [{"he": [hx, hy], "ihc": [ix, iy]}, ...]  or  {"points": [...]}
    python setup/coarse_to_fine/deskew_cli.py apply <pair> <depth>   < points.json
    python setup/coarse_to_fine/deskew_cli.py get   <pair>
    python setup/coarse_to_fine/deskew_cli.py clear <pair>

`apply` fits the affine and stores the points+coefficients. The affine is then
applied as an image warp of the moving (IHC) channel at crop time (crop_core),
so it does NOT write a displacement field. Cached FFT candidates are discarded
so higher levels recompute against the freshly-warped moving image.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

from setup.coarse_to_fine import deskew
from setup.coarse_to_fine.reg_branches import clear_lam_caches


def _clear_caches(pair_id: int) -> int:
    return clear_lam_caches(pair_id)


def apply(pair_id: int, depth: int, points: list[dict]) -> dict:
    if len(points) < 3:
        return {"error": "deskew needs at least 3 correspondence pairs"}
    for p in points:
        if "he" not in p or "ihc" not in p:
            return {"error": "each point needs 'he' and 'ihc' [x, y] entries"}

    field = deskew.fit(points)
    deskew.save(pair_id, depth, points, field)
    cleared = _clear_caches(pair_id)
    return {
        "ok": True,
        "n": len(points),
        "depth": depth,
        "affine": list(field.affine),
        "caches_cleared": cleared,
    }


def get(pair_id: int) -> dict:
    store = deskew.load(pair_id)
    if not store:
        return {"points": [], "depth": None}
    return {"points": store.get("points", []), "depth": store.get("depth")}


def clear(pair_id: int) -> dict:
    deskew.clear(pair_id)
    return {"ok": True, "cleared": pair_id}


def _read_points_stdin() -> list[dict]:
    raw = sys.stdin.read().strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        return list(data.get("points", []))
    return list(data)


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        sys.exit("Usage: deskew_cli.py <apply|get|clear> <pair> [depth]")
    command = argv[0]

    if command == "apply":
        if len(argv) < 3:
            sys.exit("Usage: deskew_cli.py apply <pair> <depth>  (points JSON on stdin)")
        result = apply(int(argv[1]), int(argv[2]), _read_points_stdin())
    elif command == "get":
        if len(argv) < 2:
            sys.exit("Usage: deskew_cli.py get <pair>")
        result = get(int(argv[1]))
    elif command == "clear":
        if len(argv) < 2:
            sys.exit("Usage: deskew_cli.py clear <pair>")
        result = clear(int(argv[1]))
    else:
        result = {"error": f"unknown command {command}"}

    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
