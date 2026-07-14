"""
Coarse-to-fine multi-level registration orchestrator (headless, Phase 1).

For each pair, loop over levels 3 -> 4 -> 5.  At each level, the moving tile
(IHC) is warped in-memory by the finished field from the coarser levels so the
FFT only measures the small residual; the residual is composed back onto the
coarse prediction (total = coarse + residual).  Candidate FFT displacements are
tau-gated against the human-anchored field (or robustly fit when no human
annotations exist yet), and the resulting smooth field is written to
data/smooth_c2f/{pair}_smooth_field.json (schema-compatible with smooth_field.py).

Human annotations (data/registration_annotations.json) are first-class: they are
loaded, always honoured, and FFT never overrides a human anchor.  The standalone
per-tile elastix/displacement.json output is never touched.

Usage:
    python setup/coarse_to_fine/run.py <pair_id> [--levels 3 4 5] [--tau 0.01] [--force]
    python setup/coarse_to_fine/run.py --pairs N [--levels 3 4 5] [--tau 0.01] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import shift as ndimage_shift

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

sys.path.insert(0, str(REPO_ROOT / "setup" / "auto-alignment"))
import align  # noqa: E402  (hyphenated dir, imported via sys.path)

from setup.coarse_to_fine import annotations
from setup.coarse_to_fine.field import (
    Candidate,
    Field,
    candidate_to_dict,
    fit_field,
    fit_gated,
    write_field_json,
)

CROPPED_DIR = conf.PROJECT_ROOT / "data" / "cropped"
CACHE_DIR = conf.PROJECT_ROOT / "data" / "c2f_cache"
DEFAULT_LEVELS = [3, 4, 5]
DEFAULT_TAU = 0.01


def _tissue_tiles(pair_id: int, level: int) -> list[str]:
    """Tile ids present under data/cropped/<pair>/d<level>/ (background tiles are absent)."""
    depth_dir = CROPPED_DIR / str(pair_id) / f"d{level}"
    if not depth_dir.is_dir():
        return []
    tiles = []
    for d in depth_dir.iterdir():
        if not d.is_dir():
            continue
        if "_" not in d.name:
            continue
        if (d / "he.png").exists() and (d / "ihc.png").exists():
            tiles.append(d.name)
    return sorted(tiles)


def _load_gray(pair_id: int, level: int, tile_loc: str, side: str) -> np.ndarray | None:
    path = CROPPED_DIR / str(pair_id) / f"d{level}" / tile_loc / f"{side}.png"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else img.astype(np.float64)


def _level_candidates(pair_id: int, level: int, coarse: Field, on_progress=None) -> list[Candidate]:
    """
    Run residual FFT for every tissue tile at `level`, warping IHC in-memory by
    the coarse field prediction, and compose total = coarse + residual.
    `on_progress(done, total)` is called after each tile if provided.
    """
    tiles = _tissue_tiles(pair_id, level)
    total = len(tiles)
    candidates: list[Candidate] = []
    for done, tile_loc in enumerate(tiles, start=1):
        he = _load_gray(pair_id, level, tile_loc, "he")
        ihc = _load_gray(pair_id, level, tile_loc, "ihc")
        if he is not None and ihc is not None:
            cdx, cdy = coarse.predict_tile_px_at(level, tile_loc)
            ihc_warp = ndimage_shift(ihc, shift=(cdy, cdx), order=1, mode="nearest")
            res = align.register_arrays(he, ihc_warp)
            candidates.append(
                Candidate(
                    level=level,
                    tile_loc=tile_loc,
                    u=cdx + res["dx"],
                    v=cdy + res["dy"],
                    psr=res["psr"],
                )
            )
        if on_progress is not None:
            on_progress(done, total)
    return candidates


def _fit_level(
    field_coarse: Field,
    candidates: list[Candidate],
    entries: list[dict],
    level: int,
    tau: float,
) -> tuple[Field, list[Candidate]]:
    """Fit the field at one level: honour human anchors <= level, tau-gate FFT soft points.

    Tiles explicitly marked as `exclude` are removed from the candidate set so they
    do not influence the fit.
    """
    excluded = {
        e["tile_loc"]
        for e in entries
        if int(e["level"]) == level and e.get("type") == "exclude"
    }
    human_anchors = annotations.to_anchors(entries, up_to_level=level)
    kept_candidates = [c for c in candidates if c.tile_loc not in excluded]
    return fit_gated(human_anchors, kept_candidates, tau)


def compute_candidates(
    pair_id: int,
    target_depth: int,
    levels: list[int],
    tau: float,
    on_progress=None,
) -> tuple[list[Candidate], Field]:
    """
    Build the coarse field by running all c2f levels below `target_depth`, then
    compute (and return) the target level's candidate residual displacements.
    `on_progress(done, total)` reports progress over the target level's tiles.
    """
    entries = annotations.load(pair_id)
    field = fit_field(annotations.to_anchors(entries, up_to_level=min(levels) - 1))

    for level in [lv for lv in levels if lv < target_depth]:
        cands = _level_candidates(pair_id, level, field)
        if cands:
            field, _ = _fit_level(field, cands, entries, level, tau)

    target_cands = _level_candidates(pair_id, target_depth, field, on_progress=on_progress)
    return target_cands, field


def cache_candidates(pair_id: int, target_depth: int, levels: list[int], tau: float) -> Path:
    """Compute candidates for one pair+depth and cache them, emitting progress lines."""
    def _progress(done: int, total: int) -> None:
        print(f"done={done} total={total}", flush=True)

    candidates, _ = compute_candidates(pair_id, target_depth, levels, tau, on_progress=_progress)
    payload = {
        "pair_id": pair_id,
        "depth": target_depth,
        "levels": levels,
        "candidates": [candidate_to_dict(c) for c in candidates],
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{pair_id}_d{target_depth}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"pair {pair_id}  depth {target_depth}: cached {len(candidates)} candidates -> {out_path.name}")
    return out_path


def process_pair(pair_id: int, levels: list[int], tau: float) -> Field | None:
    entries = annotations.load(pair_id)  # human actions only (empty in Phase 1)

    # Coarse field going into the first level = human anchors below it, else identity.
    field = fit_field(annotations.to_anchors(entries, up_to_level=levels[0] - 1))

    total_kept = 0
    total_seen = 0
    for level in levels:
        candidates = _level_candidates(pair_id, level, field)
        if not candidates:
            print(f"pair {pair_id}  level {level}: no tissue tiles, skipping")
            continue
        total_seen += len(candidates)

        field, kept = _fit_level(field, candidates, entries, level, tau)

        total_kept += len(kept)
        print(
            f"pair {pair_id}  level {level}: {len(candidates)} tiles, "
            f"{len(kept)} kept, {len(candidates) - len(kept)} rejected"
        )

    if total_seen == 0:
        print(f"pair {pair_id}: nothing to fit, skipping")
        return None

    meta = {
        "levels": levels,
        "tau": tau,
        "n_human_anchors": len(entries),
        "n_kept": total_kept,
        "n_seen": total_seen,
    }
    out_path = write_field_json(pair_id, field, meta=meta)
    print(f"pair {pair_id}: saved {out_path.name}  ({total_kept}/{total_seen} FFT tiles kept)")
    return field


def _all_pairs() -> list[int]:
    if not CROPPED_DIR.is_dir():
        return []
    return sorted(int(p.name) for p in CROPPED_DIR.iterdir() if p.is_dir() and p.name.isdigit())


def _pair_done(pair_id: int) -> bool:
    return (conf.PROJECT_ROOT / "data" / "smooth_c2f" / f"{pair_id}_smooth_field.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Coarse-to-fine registration orchestrator")
    parser.add_argument("pair_id", type=int, nargs="?", help="single pair id")
    parser.add_argument("--pairs", type=int, help="process next N pairs without a c2f field")
    parser.add_argument("--cache-depth", type=int, help="compute+cache candidates for one depth (UI)")
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    levels = sorted(args.levels)

    if args.cache_depth is not None:
        if args.pair_id is None:
            parser.error("--cache-depth requires a pair_id")
        cache_candidates(args.pair_id, args.cache_depth, levels, args.tau)
        return

    if args.pairs is not None:
        queued = [p for p in _all_pairs() if args.force or not _pair_done(p)]
        batch = queued[: args.pairs]
        if not batch:
            print("Nothing to process — all pairs already have a c2f field (use --force to rerun).")
            return
        for pid in batch:
            process_pair(pid, levels, args.tau)
        return

    if args.pair_id is None:
        parser.error("provide a pair_id or --pairs N")

    process_pair(args.pair_id, levels, args.tau)


if __name__ == "__main__":
    main()
