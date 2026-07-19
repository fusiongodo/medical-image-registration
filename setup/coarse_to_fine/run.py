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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

sys.path.insert(0, str(REPO_ROOT / "setup" / "auto-alignment"))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))
import align  # noqa: E402  (hyphenated dir, imported via sys.path)
import crop_core  # noqa: E402
import tile_metrics  # noqa: E402

from setup.coarse_to_fine import annotations
from setup.coarse_to_fine.identity import pair_fingerprint
from setup.coarse_to_fine.field import (
    Candidate,
    Field,
    candidate_from_dict,
    fit_field,
    fit_gated,
    write_field_json,
)

CACHE_DIR = conf.PROJECT_ROOT / "data" / "c2f_cache"
DEFAULT_LEVELS = [0, 1, 2, 3, 4, 5]
DEFAULT_TAU = 0.01


def _tissue_tiles(pair_id: int, level: int) -> list[str]:
    """Tile ids at this quadtree level, cropped live from the raw WSI (mask-filtered)."""
    return crop_core.tissue_tiles(pair_id, level)["tiles"]


def _level_records(
    pair_id: int,
    level: int,
    coarse: Field,
    with_metrics: bool = False,
    on_progress=None,
) -> list[dict]:
    """
    Residual FFT for every tissue tile at `level`. IHC is recropped from the raw
    WSI at the coarse field prediction (full intersection) before the FFT, and
    the residual is composed back onto that prediction (total = coarse + residual).
    Records are `{tile_loc, u, v, psr[, delta_px, by_patch]}` (metrics only when
    `with_metrics`). `on_progress(done, total)` is called after each tile.
    """
    tiles = _tissue_tiles(pair_id, level)
    total = len(tiles)
    records: list[dict] = []
    for done, tile_loc in enumerate(tiles, start=1):
        x, y = (int(p) for p in tile_loc.split("_"))
        cdx, cdy = coarse.predict_tile_px_at(level, tile_loc)
        he = crop_core.crop_gray(pair_id, level, x, y, "he")
        ihc_base = crop_core.crop_gray(pair_id, level, x, y, "ihc", dx=cdx, dy=cdy)
        res = align.register_arrays(he, ihc_base)
        u, v = cdx + res["dx"], cdy + res["dy"]
        record = {"tile_loc": tile_loc, "u": u, "v": v, "psr": res["psr"]}
        if with_metrics:
            ihc_auto = crop_core.crop_gray(pair_id, level, x, y, "ihc", dx=u, dy=v)
            record.update(tile_metrics.tile_metrics(he, ihc_base, ihc_auto, u, v))
        records.append(record)
        if on_progress is not None:
            on_progress(done, total)
    return records


def _level_candidates(pair_id: int, level: int, coarse: Field, on_progress=None) -> list[Candidate]:
    """Candidate objects for fitting/replay (no metrics)."""
    return [
        candidate_from_dict(level, r)
        for r in _level_records(pair_id, level, coarse, with_metrics=False, on_progress=on_progress)
    ]


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
) -> tuple[list[dict], Field]:
    """
    Build the coarse field by replaying all c2f levels below `target_depth`
    (which reproduces the saved previous-level field), then compute (and return)
    the target level's candidate records including per-tile LNCC metrics.
    `on_progress(done, total)` reports progress over the target level's tiles.
    """
    entries = annotations.load(pair_id)
    field = fit_field(annotations.to_anchors(entries, up_to_level=min(levels) - 1))

    for level in [lv for lv in levels if lv < target_depth]:
        cands = _level_candidates(pair_id, level, field)
        if cands:
            field, _ = _fit_level(field, cands, entries, level, tau)

    records = _level_records(pair_id, target_depth, field, with_metrics=True, on_progress=on_progress)
    return records, field


def cache_candidates(pair_id: int, target_depth: int, levels: list[int], tau: float) -> Path:
    """Compute candidate records (with metrics) for one pair+depth and cache them."""
    def _progress(done: int, total: int) -> None:
        print(f"done={done} total={total}", flush=True)

    records, _ = compute_candidates(pair_id, target_depth, levels, tau, on_progress=_progress)
    payload = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "depth": target_depth,
        "levels": levels,
        "candidates": records,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{pair_id}_d{target_depth}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"pair {pair_id}  depth {target_depth}: cached {len(records)} candidates -> {out_path.name}")
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
    return list(range(crop_core.num_pairs()))


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
