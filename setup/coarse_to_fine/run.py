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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

sys.path.insert(0, str(REPO_ROOT / "setup" / "auto-alignment"))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))
import align  # noqa: E402  (hyphenated dir, imported via sys.path)
import crop_core  # noqa: E402
import tile_metrics  # noqa: E402

from setup.coarse_to_fine import annotations, masks
from setup.coarse_to_fine.identity import pair_fingerprint
from setup.coarse_to_fine.field import (
    Candidate,
    Field,
    candidate_from_dict,
    fit_field,
    fit_gated,
    write_field_json,
)
from setup.coarse_to_fine.reg_branches import (
    DEFAULT_FIELD_ESTIMATOR,
    DEFAULT_LAM,
    DEFAULT_WENDLAND_EPS,
    FIELD_ESTIMATORS,
    LAMS,
    cache_path,
    normalize_estimator,
    normalize_lam,
)

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
    lam: str | None = None,
) -> list[dict]:
    """
    Residual LAM for every tissue tile at `level`. IHC is recropped from the raw
    WSI at the coarse field prediction (full intersection) before the LAM, and
    the residual is composed back onto that prediction (total = coarse + residual).

    FFT → align.register_arrays. superpoint_glue → mean inlier match displacement.
    Records are `{tile_loc, u, v, psr[, delta_px, by_patch, matches]}`.
    """
    lam = normalize_lam(lam)
    tiles = _tissue_tiles(pair_id, level)
    total = len(tiles)
    records: list[dict] = []

    sp_ctx = None
    load_models_s = 0.0
    timing_sum = {"sp_he_s": 0.0, "sp_ihc_s": 0.0, "sp_s": 0.0, "lg_s": 0.0, "crop_s": 0.0}
    if lam == "superpoint_glue":
        from setup.coarse_to_fine import lam_sp_lg

        print("stage=load_models", flush=True)
        t_load = time.perf_counter()
        extractor, matcher, device, hp = lam_sp_lg.build_models()
        load_models_s = time.perf_counter() - t_load
        print(f"stage=models_ready device={device} load_s={load_models_s:.3f}", flush=True)
        sp_ctx = (extractor, matcher, device, hp)

    for done, tile_loc in enumerate(tiles, start=1):
        x, y = (int(p) for p in tile_loc.split("_"))
        cdx, cdy = coarse.predict_tile_px_at(level, tile_loc)
        t_crop = time.perf_counter()
        he = crop_core.crop_gray(pair_id, level, x, y, "he")
        ihc_base = crop_core.crop_gray(pair_id, level, x, y, "ihc", dx=cdx, dy=cdy)
        crop_s = time.perf_counter() - t_crop

        if lam == "superpoint_glue":
            assert sp_ctx is not None
            extractor, matcher, device, hp = sp_ctx

            def _stage(name: str, _tile=tile_loc, _i=done, _total=total) -> None:
                print(
                    f"stage={name} tile={_tile} i={_i} total={_total}",
                    flush=True,
                )

            res = lam_sp_lg.match_tile_residual(
                he,
                ihc_base,
                extractor,
                matcher,
                device,
                hp,
                resize=None,
                on_stage=_stage,
            )
            u, v = cdx + res["dx"], cdy + res["dy"]
            t = res.get("timing") or {}
            for k in ("sp_he_s", "sp_ihc_s", "sp_s", "lg_s"):
                timing_sum[k] += float(t.get(k, 0.0))
            timing_sum["crop_s"] += crop_s
            record = {
                "tile_loc": tile_loc,
                "u": u,
                "v": v,
                "psr": res["psr"],
                "delta_px": (u ** 2 + v ** 2) ** 0.5,
                "n_matches": res["n_matches"],
                "n_inliers": res["n_inliers"],
                "matches": res["matches"],
            }
        else:
            res = align.register_arrays(he, ihc_base)
            u, v = cdx + res["dx"], cdy + res["dy"]
            record = {
                "tile_loc": tile_loc,
                "u": u,
                "v": v,
                "psr": res["psr"],
                "delta_px": (u ** 2 + v ** 2) ** 0.5,
            }

        if with_metrics:
            ihc_auto = crop_core.crop_gray(pair_id, level, x, y, "ihc", dx=u, dy=v)
            record.update(tile_metrics.tile_metrics(he, ihc_base, ihc_auto, u, v))
        records.append(record)
        if on_progress is not None:
            on_progress(done, total)

    if lam == "superpoint_glue" and total > 0:
        sp = timing_sum["sp_s"]
        lg = timing_sum["lg_s"]
        crop = timing_sum["crop_s"]
        infer = sp + lg
        n = total
        print(
            f"timing level={level} tiles={n} "
            f"load_models={load_models_s:.3f}s "
            f"crop={crop:.3f}s "
            f"superpoint={sp:.3f}s (he={timing_sum['sp_he_s']:.3f}s ihc={timing_sum['sp_ihc_s']:.3f}s) "
            f"lightglue={lg:.3f}s "
            f"infer={infer:.3f}s "
            f"per_tile_sp={sp / n:.3f}s per_tile_lg={lg / n:.3f}s",
            flush=True,
        )
    return records


def _level_candidates(
    pair_id: int,
    level: int,
    coarse: Field,
    on_progress=None,
    lam: str | None = None,
) -> list[Candidate]:
    """Candidate objects for fitting/replay (no metrics)."""
    return [
        candidate_from_dict(level, r)
        for r in _level_records(
            pair_id, level, coarse, with_metrics=False, on_progress=on_progress, lam=lam
        )
    ]


def _fit_level(
    field_coarse: Field,
    candidates: list[Candidate],
    entries: list[dict],
    level: int,
    tau: float,
    masked: "set[str] | None" = None,
    field_estimator: str | None = None,
) -> tuple[Field, list[Candidate]]:
    """Fit the field at one level: honour human anchors <= level, tau-gate soft points.

    Tiles explicitly marked as `exclude`, or masked out (propagated by index),
    are removed from the candidate set so they do not influence the fit.
    """
    dropped = {
        e["tile_loc"]
        for e in entries
        if int(e["level"]) == level and e.get("type") == "exclude"
    }
    if masked:
        dropped |= masked
    human_anchors = annotations.to_anchors(entries, up_to_level=level)
    kept_candidates = [c for c in candidates if c.tile_loc not in dropped]
    return fit_gated(
        human_anchors, kept_candidates, tau, field_estimator=field_estimator
    )


def _load_level_cache(
    pair_id: int,
    level: int,
    lam: str | None = None,
    field_estimator: str | None = None,
    wendland_eps: float | None = None,
) -> list[Candidate] | None:
    path = cache_path(
        pair_id,
        level,
        lam,
        field_estimator=field_estimator,
        wendland_eps=wendland_eps,
    )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return [candidate_from_dict(level, d) for d in payload.get("candidates", [])]
    except Exception:
        return None


def write_candidate_cache(
    pair_id: int,
    depth: int,
    records: list[dict],
    *,
    lam: str | None = None,
    field_estimator: str | None = None,
    levels: list[int] | None = None,
    wendland_eps: float | None = None,
) -> Path:
    lam = normalize_lam(lam)
    est = normalize_estimator(field_estimator)
    payload = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "depth": depth,
        "levels": levels,
        "lam": lam,
        "field_estimator": est,
        "candidates": records,
    }
    if est == "wendland":
        eps = DEFAULT_WENDLAND_EPS if wendland_eps is None else float(wendland_eps)
        payload["wendland_eps"] = eps
    else:
        eps = wendland_eps
    out_path = cache_path(
        pair_id, depth, lam, field_estimator=est, wendland_eps=eps
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return out_path


def _coarse_field(
    pair_id: int,
    target_depth: int,
    levels: list[int],
    tau: float,
    lam: str | None = None,
    field_estimator: str | None = None,
) -> Field:
    lam = normalize_lam(lam)
    est = normalize_estimator(field_estimator)
    entries = annotations.load(pair_id)
    mask_entries = masks.load(pair_id)
    field = fit_field(
        annotations.to_anchors(entries, up_to_level=min(levels) - 1),
        field_estimator=est,
    )
    for level in [lv for lv in levels if lv < target_depth]:
        cands = _load_level_cache(pair_id, level, lam=lam, field_estimator=est)
        if cands is None:
            cands = _level_candidates(pair_id, level, field, lam=lam)
        if cands:
            masked = masks.masked_at(mask_entries, level, [c.tile_loc for c in cands])
            field, _ = _fit_level(
                field, cands, entries, level, tau, masked=masked, field_estimator=est
            )
    return field


def compute_candidates(
    pair_id: int,
    target_depth: int,
    levels: list[int],
    tau: float,
    with_metrics: bool = False,
    on_progress=None,
    lam: str | None = None,
    field_estimator: str | None = None,
) -> tuple[list[dict], Field]:
    field = _coarse_field(
        pair_id, target_depth, levels, tau, lam=lam, field_estimator=field_estimator
    )
    records = _level_records(
        pair_id,
        target_depth,
        field,
        with_metrics=with_metrics,
        on_progress=on_progress,
        lam=lam,
    )
    return records, field


def cache_candidates(
    pair_id: int,
    target_depth: int,
    levels: list[int],
    tau: float,
    lam: str | None = None,
    field_estimator: str | None = None,
) -> Path:
    lam = normalize_lam(lam)
    est = normalize_estimator(field_estimator)
    print(
        f"stage=start pair={pair_id} depth={target_depth} lam={lam} estimator={est}",
        flush=True,
    )

    def _progress(done: int, total: int) -> None:
        print(f"done={done} total={total}", flush=True)

    records, _ = compute_candidates(
        pair_id,
        target_depth,
        levels,
        tau,
        with_metrics=False,
        on_progress=_progress,
        lam=lam,
        field_estimator=est,
    )
    out_path = write_candidate_cache(
        pair_id,
        target_depth,
        records,
        lam=lam,
        field_estimator=est,
        levels=levels,
    )
    print(
        f"pair {pair_id}  depth {target_depth}  lam {lam} estimator={est}: "
        f"cached {len(records)} candidates -> {out_path.relative_to(conf.PROJECT_ROOT)}"
    )
    return out_path


def augment_metrics(
    pair_id: int,
    target_depth: int,
    levels: list[int],
    tau: float,
    lam: str | None = None,
    field_estimator: str | None = None,
) -> Path:
    lam = normalize_lam(lam)
    est = normalize_estimator(field_estimator)
    out_path = cache_path(pair_id, target_depth, lam, field_estimator=est)
    if not out_path.exists():
        raise SystemExit(
            f"no cached candidates for pair {pair_id} depth {target_depth} "
            f"lam {lam} estimator={est}; compute candidates first"
        )
    payload = json.loads(out_path.read_text())
    records = payload.get("candidates", [])
    total = len(records)

    field = _coarse_field(
        pair_id, target_depth, levels, tau, lam=lam, field_estimator=est
    )
    for done, rec in enumerate(records, start=1):
        tile_loc = rec["tile_loc"]
        x, y = (int(p) for p in tile_loc.split("_"))
        cdx, cdy = field.predict_tile_px_at(target_depth, tile_loc)
        u, v = rec["u"], rec["v"]
        he = crop_core.crop_gray(pair_id, target_depth, x, y, "he")
        ihc_base = crop_core.crop_gray(pair_id, target_depth, x, y, "ihc", dx=cdx, dy=cdy)
        ihc_auto = crop_core.crop_gray(pair_id, target_depth, x, y, "ihc", dx=u, dy=v)
        rec.update(tile_metrics.tile_metrics(he, ihc_base, ihc_auto, u, v))
        print(f"done={done} total={total}", flush=True)

    payload["candidates"] = records
    payload["lam"] = lam
    payload["field_estimator"] = est
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"pair {pair_id}  depth {target_depth}  lam {lam} estimator={est}: "
        f"LNCC metrics for {total} candidates -> {out_path.name}"
    )
    return out_path


def process_pair(
    pair_id: int,
    levels: list[int],
    tau: float,
    lam: str | None = None,
    field_estimator: str | None = None,
) -> Field | None:
    lam = normalize_lam(lam)
    est = normalize_estimator(field_estimator)
    entries = annotations.load(pair_id)

    field = fit_field(
        annotations.to_anchors(entries, up_to_level=levels[0] - 1),
        field_estimator=est,
    )

    total_kept = 0
    total_seen = 0
    for level in levels:
        candidates = _level_candidates(pair_id, level, field, lam=lam)
        if not candidates:
            print(f"pair {pair_id}  level {level}: no tissue tiles, skipping")
            continue
        total_seen += len(candidates)

        field, kept = _fit_level(
            field, candidates, entries, level, tau, field_estimator=est
        )

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
        "lam": lam,
        "field_estimator": est,
        "n_human_anchors": len(entries),
        "n_kept": total_kept,
        "n_seen": total_seen,
    }
    out_path = write_field_json(pair_id, field, meta=meta)
    print(f"pair {pair_id}: saved {out_path.name}  ({total_kept}/{total_seen} tiles kept)")
    return field


def _all_pairs() -> list[int]:
    return list(range(crop_core.num_pairs()))


def _pair_done(pair_id: int) -> bool:
    return (conf.PROJECT_ROOT / "data" / "smooth_c2f" / f"{pair_id}_smooth_field.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Coarse-to-fine registration orchestrator")
    parser.add_argument("pair_id", type=int, nargs="?", help="single pair id")
    parser.add_argument("--pairs", type=int, help="process next N pairs without a c2f field")
    parser.add_argument("--cache-depth", type=int, help="compute+cache LAM candidates for one depth (UI)")
    parser.add_argument("--metrics-depth", type=int, help="add LNCC metrics to cached candidates for one depth (UI)")
    parser.add_argument("--lam", default=DEFAULT_LAM, choices=LAMS)
    parser.add_argument(
        "--estimator",
        default=DEFAULT_FIELD_ESTIMATOR,
        choices=FIELD_ESTIMATORS,
    )
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    levels = sorted(args.levels)
    lam = normalize_lam(args.lam)
    est = normalize_estimator(args.estimator)

    if args.cache_depth is not None:
        if args.pair_id is None:
            parser.error("--cache-depth requires a pair_id")
        cache_candidates(
            args.pair_id, args.cache_depth, levels, args.tau, lam=lam, field_estimator=est
        )
        return

    if args.metrics_depth is not None:
        if args.pair_id is None:
            parser.error("--metrics-depth requires a pair_id")
        augment_metrics(
            args.pair_id, args.metrics_depth, levels, args.tau, lam=lam, field_estimator=est
        )
        return

    if args.pairs is not None:
        queued = [p for p in _all_pairs() if args.force or not _pair_done(p)]
        batch = queued[: args.pairs]
        if not batch:
            print("Nothing to process — all pairs already have a c2f field (use --force to rerun).")
            return
        for pid in batch:
            process_pair(pid, levels, args.tau, lam=lam, field_estimator=est)
        return

    if args.pair_id is None:
        parser.error("provide a pair_id or --pairs N")

    process_pair(args.pair_id, levels, args.tau, lam=lam, field_estimator=est)


if __name__ == "__main__":
    main()
