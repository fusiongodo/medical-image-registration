"""
Fast tau refit for the coarse-to-fine UI.

Loads the cached candidate residual displacements for one pair+depth (written by
run.py --cache-depth) together with the durable human annotations, refits the
smooth field at the requested tau (human anchors honoured, remaining FFT
candidates tau-gated), and prints a per-tile agreement report as JSON to stdout.

Compared to the earlier version, this script also computes the previous-level
(prior) field so the UI can show the "included" candidate displacement
(u, v = prior-field + FFT residual) alongside the "excluded" prior-field
fallback.

Usage:
    python setup/coarse_to_fine/refit_cli.py <pair_id> <depth> <tau> [--save]

With --save the fitted field is also written to data/smooth_c2f/{pair}_smooth_field.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

from setup.coarse_to_fine import annotations
from setup.coarse_to_fine.field import (
    Candidate,
    Field,
    candidate_from_dict,
    fit_field,
    fit_gated,
    residuals,
    write_field_json,
)
from setup.coarse_to_fine.run import _level_candidates

CACHE_DIR = conf.PROJECT_ROOT / "data" / "c2f_cache"


def _load_level_cache(pair_id: int, level: int) -> list[Candidate] | None:
    path = CACHE_DIR / f"{pair_id}_d{level}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return [candidate_from_dict(level, d) for d in payload.get("candidates", [])]
    except Exception:
        return None


def _compute_prior_field(
    pair_id: int,
    depth: int,
    tau: float,
    levels: list[int],
) -> Field:
    """
    Replay all coarser levels (< depth) to obtain the prior field that was used
    to warp IHC before measuring the current depth's candidates.

    Lower-level caches are used when available; missing levels are recomputed on
    the fly.
    """
    entries = annotations.load(pair_id)
    min_level = min(levels)
    field = fit_field(annotations.to_anchors(entries, up_to_level=min_level - 1))

    for level in sorted(lv for lv in levels if lv < depth):
        cands = _load_level_cache(pair_id, level)
        if cands is None:
            cands = _level_candidates(pair_id, level, field)
        if not cands:
            continue
        human_anchors = annotations.to_anchors(entries, up_to_level=level)
        field, _ = fit_gated(human_anchors, cands, tau)

    return field


def refit(pair_id: int, depth: int, tau: float, save: bool) -> dict:
    cache_path = CACHE_DIR / f"{pair_id}_d{depth}.json"
    if not cache_path.exists():
        return {"error": f"no cached candidates for pair {pair_id} depth {depth}"}

    payload = json.loads(cache_path.read_text())
    levels = payload.get("levels", [3, 4, 5])
    candidates = [candidate_from_dict(depth, d) for d in payload.get("candidates", [])]

    entries = annotations.load(pair_id)
    human_anchors = annotations.to_anchors(entries, up_to_level=depth)
    annotated = {e["tile_loc"]: e["type"] for e in entries if int(e["level"]) == depth}
    excluded_locs = {
        e["tile_loc"]
        for e in entries
        if int(e["level"]) == depth and e.get("type") == "exclude"
    }
    n_human = len(human_anchors)

    if not candidates and not human_anchors:
        return {
            "tau": tau, "kept": 0, "rejected": 0, "n_human": 0,
            "mean_residual": 0.0, "tiles": [],
        }

    prior_field = _compute_prior_field(pair_id, depth, tau, levels)

    fit_candidates = [c for c in candidates if c.tile_loc not in excluded_locs]
    field, _ = fit_gated(human_anchors, fit_candidates, tau)
    devs = residuals(candidates, field)

    tiles = []
    for cand, dev in zip(candidates, devs):
        dx, dy = field.predict_tile_px_at(depth, cand.tile_loc)
        prior_dx, prior_dy = prior_field.predict_tile_px_at(depth, cand.tile_loc)
        ann = annotated.get(cand.tile_loc)
        is_excluded = ann == "exclude"
        tiles.append({
            "tile_loc": cand.tile_loc,
            "psr": cand.psr,
            "residual": dev,
            "kept": not is_excluded and (ann is not None or dev <= tau),
            "excluded": is_excluded,
            "annotated": ann,
            "dx": dx,
            "dy": dy,
            "ux": cand.u,
            "uy": cand.v,
            "prior_dx": prior_dx,
            "prior_dy": prior_dy,
        })

    n_kept = sum(1 for t in tiles if t["kept"])
    n_excluded = sum(1 for t in tiles if t["excluded"])
    result = {
        "tau": tau,
        "kept": n_kept,
        "rejected": len(tiles) - n_kept - n_excluded,
        "excluded": n_excluded,
        "n_human": n_human,
        "mean_residual": float(sum(devs) / len(devs)) if devs else 0.0,
        "tiles": tiles,
    }

    if save:
        meta = {
            "levels": levels,
            "tau": tau,
            "n_kept": n_kept,
            "n_seen": len(tiles),
            "n_human": n_human,
        }
        write_field_json(pair_id, field, meta=meta)
        result["saved"] = True

    return result


def main() -> None:
    argv = sys.argv[1:]
    save = "--save" in argv
    argv = [a for a in argv if a != "--save"]
    if len(argv) < 3:
        sys.exit("Usage: refit_cli.py <pair_id> <depth> <tau> [--save]")
    pair_id, depth, tau = int(argv[0]), int(argv[1]), float(argv[2])
    print(json.dumps(refit(pair_id, depth, tau, save), separators=(",", ":")))


if __name__ == "__main__":
    main()
