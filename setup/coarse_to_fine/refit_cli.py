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

from setup.coarse_to_fine import annotations, masks
from setup.coarse_to_fine.field import (
    Candidate,
    Field,
    candidate_from_dict,
    fit_field,
    fit_gated,
    residuals,
    tau_for_keep,
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
    mask_entries: list[dict],
) -> Field:
    """
    Replay all coarser levels (< depth) to obtain the prior field that was used
    to warp IHC before measuring the current depth's candidates.

    Lower-level caches are used when available; missing levels are recomputed on
    the fly. Masked tiles (propagated by index) are dropped so the prior field
    matches the saved field. A global deskew, when present, is baked into the
    moving crops upstream (crop_core), so it needs no handling here.
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
        masked = masks.masked_at(mask_entries, level, [c.tile_loc for c in cands])
        cands = [c for c in cands if c.tile_loc not in masked]
        human_anchors = annotations.to_anchors(entries, up_to_level=level)
        field, _ = fit_gated(human_anchors, cands, tau)

    return field


def refit(pair_id: int, depth: int, tau: float, save: bool, keep: float | None = None) -> dict:
    cache_path = CACHE_DIR / f"{pair_id}_d{depth}.json"
    if not cache_path.exists():
        return {"error": f"no cached candidates for pair {pair_id} depth {depth}"}

    payload = json.loads(cache_path.read_text())
    levels = payload.get("levels", [0, 1, 2, 3, 4, 5])
    candidates = [candidate_from_dict(depth, d) for d in payload.get("candidates", [])]

    entries = annotations.load(pair_id)
    mask_entries = masks.load(pair_id)
    masked_locs = masks.masked_at(mask_entries, depth, [c.tile_loc for c in candidates])
    human_anchors = annotations.to_anchors(entries, up_to_level=depth)
    annotated = {e["tile_loc"]: e["type"] for e in entries if int(e["level"]) == depth}
    # Human-set displacement (approve keeps the FFT pick, correct overrides it) in
    # this level's tile-pixel units, so the UI can show the actually-chosen vector
    # for annotated tiles instead of the stale FFT candidate.
    human_uv = {
        e["tile_loc"]: (float(e["disp"]["u"]), float(e["disp"]["v"]))
        for e in entries
        if int(e["level"]) == depth and e.get("type") in ("approve", "correct")
    }
    excluded_locs = {
        e["tile_loc"]
        for e in entries
        if int(e["level"]) == depth and e.get("type") == "exclude"
    }
    n_human = len(human_anchors)

    if not candidates and not human_anchors:
        return {
            "tau": tau, "keep": keep, "kept": 0, "rejected": 0, "n_human": 0,
            "mean_residual": 0.0, "tiles": [],
        }

    fit_candidates = [
        c for c in candidates
        if c.tile_loc not in excluded_locs and c.tile_loc not in masked_locs
    ]
    if keep is not None:
        # the keep-fraction is measured over auto tiles only: human-annotated
        # (approve/correct/exclude) tiles are always decided by the user
        auto_candidates = [c for c in fit_candidates if c.tile_loc not in annotated]
        tau = tau_for_keep(human_anchors, auto_candidates, keep)

    prior_field = _compute_prior_field(pair_id, depth, tau, levels, mask_entries)

    field, _ = fit_gated(human_anchors, fit_candidates, tau)
    # Judge residuals/kept against the same reference fit_gated tau-gates against
    # (human-only field when anchors exist), so the reported kept set matches the
    # tiles that actually shape the spline and the keep-fraction is exact.
    gate_field = fit_field(human_anchors) if human_anchors else field
    devs = residuals(candidates, gate_field)

    tiles = []
    for cand, dev in zip(candidates, devs):
        dx, dy = field.predict_tile_px_at(depth, cand.tile_loc)
        prior_dx, prior_dy = prior_field.predict_tile_px_at(depth, cand.tile_loc)
        ann = annotated.get(cand.tile_loc)
        is_excluded = ann == "exclude"
        is_masked = cand.tile_loc in masked_locs
        ann_u, ann_v = human_uv.get(cand.tile_loc, (None, None))
        tiles.append({
            "tile_loc": cand.tile_loc,
            "psr": cand.psr,
            "residual": dev,
            "kept": not is_excluded and not is_masked and (ann is not None or dev <= tau),
            "excluded": is_excluded,
            "masked": is_masked,
            "annotated": ann,
            "dx": dx,
            "dy": dy,
            "ux": cand.u,
            "uy": cand.v,
            "ann_u": ann_u,
            "ann_v": ann_v,
            "prior_dx": prior_dx,
            "prior_dy": prior_dy,
        })

    n_kept = sum(1 for t in tiles if t["kept"])
    n_excluded = sum(1 for t in tiles if t["excluded"] and not t["masked"])
    n_masked = sum(1 for t in tiles if t["masked"])
    result = {
        "tau": tau,
        "keep": keep,
        "kept": n_kept,
        "rejected": len(tiles) - n_kept - n_excluded - n_masked,
        "excluded": n_excluded,
        "masked": n_masked,
        "n_human": n_human,
        "mean_residual": float(sum(devs) / len(devs)) if devs else 0.0,
        "tiles": tiles,
    }

    if save:
        meta = {
            "levels": levels,
            "tau": tau,
            "keep": keep,
            "saved_depth": depth,
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

    keep: float | None = None
    if "--keep" in argv:
        i = argv.index("--keep")
        keep = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    if len(argv) < 3:
        sys.exit("Usage: refit_cli.py <pair_id> <depth> <tau> [--keep <fraction>] [--save]")
    pair_id, depth, tau = int(argv[0]), int(argv[1]), float(argv[2])
    print(json.dumps(refit(pair_id, depth, tau, save, keep), separators=(",", ":")))


if __name__ == "__main__":
    main()
