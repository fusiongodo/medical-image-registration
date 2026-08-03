"""
Refinement-aware FFT recompute for red (rejected / excluded) tiles.

The candidate FFT run at compute time recrops each tile's IHC at the *previous*
level's field and keeps only the single top correlation peak. When that peak
points the wrong way the tile ends up above tau (red). This pass revisits every
red tile at pair+depth using the *current* refinement field as prior:

  1. build the same gate field refit judges red/green against
     (human-only field when anchors exist, else the fitted included field),
  2. recrop the tile's IHC at that field's prediction (field-aware crop),
  3. run a multi-peak (NMS) phase correlation on the recropped tile,
  4. choose the highest-PSR peak whose composed displacement still falls
     within tau of the field prediction.

The winning (u, v, psr) overwrites that tile's candidate in
data/c2f_cache/{pair}_d{depth}.json (stale by_patch dropped). An auto-rejected
tile is additionally pinned as an `approve` anchor so it stays green in both gate
modes -- under exclude-% the kept fraction is fixed, so a rewritten candidate
alone would only swap which tile is red. Tiles with no in-tau peak are left
unchanged (stay red). Manually excluded tiles are recomputed too but keep their
exclude vote; the user clears it individually to fold in the field-consistent peak.

Usage:
    python setup/coarse_to_fine/resolve_cli.py <pair_id> <depth> <tau> [--keep <f>] [--n <peaks>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "auto-alignment"))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))
import conf

import align
import crop_core

from setup.coarse_to_fine import annotations, masks
from setup.coarse_to_fine.field import (
    Candidate,
    candidate_from_dict,
    fit_field,
    fit_gated,
    psr_to_conf,
    residuals,
    tau_for_keep,
)

CACHE_DIR = conf.PROJECT_ROOT / "data" / "c2f_cache"
N_PEAKS = 5


def _best_in_tau_peak(
    pair_id: int, depth: int, tile_loc: str, pred_dx: float, pred_dy: float,
    gate_field, tau: float, n_peaks: int,
) -> "tuple[float, float, float, float] | None":
    """
    Recrop the tile's IHC at (pred_dx, pred_dy), run multi-peak FFT, and return the
    (u, v, psr, residual) of the highest-PSR peak within tau of the field, or None.
    """
    x, y = (int(p) for p in tile_loc.split("_"))
    he = crop_core.crop_gray(pair_id, depth, x, y, "he")
    ihc = crop_core.crop_gray(pair_id, depth, x, y, "ihc", dx=pred_dx, dy=pred_dy)
    peaks = align.register_arrays_multi(he, ihc, n_peaks=n_peaks)

    best: "tuple[float, float, float, float] | None" = None
    for pk in peaks:
        u, v = pred_dx + pk["dx"], pred_dy + pk["dy"]
        cand = Candidate(level=depth, tile_loc=tile_loc, u=u, v=v, psr=pk["psr"])
        res = residuals([cand], gate_field)[0]
        if res <= tau and (best is None or pk["psr"] > best[2]):
            best = (u, v, pk["psr"], res)
    return best


def resolve(
    pair_id: int,
    depth: int,
    tau: float,
    keep: "float | None" = None,
    n_peaks: int = N_PEAKS,
    field_estimator: str | None = None,
) -> dict:
    cache_path = CACHE_DIR / f"{pair_id}_d{depth}.json"
    if not cache_path.exists():
        return {"error": f"no cached candidates for pair {pair_id} depth {depth}"}

    payload = json.loads(cache_path.read_text())
    raw_by_loc = {d["tile_loc"]: d for d in payload.get("candidates", [])}
    candidates = [candidate_from_dict(depth, d) for d in payload.get("candidates", [])]

    entries = annotations.load(pair_id)
    mask_entries = masks.load(pair_id)
    masked_locs = masks.masked_at(mask_entries, depth, [c.tile_loc for c in candidates])
    human_anchors = annotations.to_anchors(entries, up_to_level=depth)
    annotated = {e["tile_loc"]: e["type"] for e in entries if int(e["level"]) == depth}
    excluded_locs = {loc for loc, t in annotated.items() if t == "exclude"}

    fit_candidates = [
        c for c in candidates
        if c.tile_loc not in excluded_locs and c.tile_loc not in masked_locs
    ]
    if keep is not None:
        auto_candidates = [c for c in fit_candidates if c.tile_loc not in annotated]
        tau = tau_for_keep(human_anchors, auto_candidates, keep, field_estimator=field_estimator)

    field, _ = fit_gated(human_anchors, fit_candidates, tau, field_estimator=field_estimator)
    gate_field = (
        fit_field(human_anchors, field_estimator=field_estimator) if human_anchors else field
    )
    devs = residuals(candidates, gate_field)

    tiles_out: list[dict] = []
    resolved = 0
    approved = 0
    new_entries: list[dict] = []
    for cand, dev in zip(candidates, devs):
        ann = annotated.get(cand.tile_loc)
        is_masked = cand.tile_loc in masked_locs
        kept = ann != "exclude" and not is_masked and (ann is not None or dev <= tau)
        if is_masked or kept:
            continue

        pred_dx, pred_dy = gate_field.predict_tile_px_at(depth, cand.tile_loc)
        best = _best_in_tau_peak(
            pair_id, depth, cand.tile_loc, pred_dx, pred_dy, gate_field, tau, n_peaks
        )
        if best is not None:
            u, v, psr, res = best
            raw = raw_by_loc[cand.tile_loc]
            raw["u"], raw["v"], raw["psr"] = u, v, psr
            raw["delta_px"] = float((u ** 2 + v ** 2) ** 0.5)
            raw.pop("by_patch", None)
            resolved += 1
            # Pin auto-rejected reds as approve anchors so they stay green in both
            # gate modes (under exclude-% the kept fraction is fixed, so a rewritten
            # candidate alone would only swap which tile is red). Manually excluded
            # tiles keep their vote; their candidate is still rewritten so clearing
            # the exclude vote later folds in the field-consistent peak.
            if ann is None:
                new_entries.append(annotations.make_entry(
                    depth, cand.tile_loc, u, v, "approve", "resolve", psr_to_conf(psr)
                ))
                approved += 1
            tiles_out.append({"tile_loc": cand.tile_loc, "resolved": True,
                              "residual": res, "psr": psr})
        else:
            tiles_out.append({"tile_loc": cand.tile_loc, "resolved": False,
                              "residual": dev})

    if resolved:
        cache_path.write_text(json.dumps(payload, separators=(",", ":")))
    if new_entries:
        annotations.save(pair_id, entries + new_entries)

    return {
        "ok": True,
        "tau": tau,
        "keep": keep,
        "tried": len(tiles_out),
        "resolved": resolved,
        "approved": approved,
        "tiles": tiles_out,
    }


def main() -> None:
    argv = sys.argv[1:]

    keep: "float | None" = None
    if "--keep" in argv:
        i = argv.index("--keep")
        keep = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    n_peaks = N_PEAKS
    if "--n" in argv:
        i = argv.index("--n")
        n_peaks = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    field_estimator: str | None = None
    if "--field-estimator" in argv:
        i = argv.index("--field-estimator")
        field_estimator = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    if len(argv) < 3:
        sys.exit(
            "Usage: resolve_cli.py <pair_id> <depth> <tau> "
            "[--keep <fraction>] [--n <peaks>] [--field-estimator tps|wendland|bspline]"
        )
    pair_id, depth, tau = int(argv[0]), int(argv[1]), float(argv[2])
    print(
        json.dumps(
            resolve(pair_id, depth, tau, keep, n_peaks, field_estimator=field_estimator),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
