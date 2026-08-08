"""
Eval batch CLI: create / list / run LAM × field-estimator grids.

Usage:
  python setup/coarse_to_fine/eval_batch_cli.py list
  python setup/coarse_to_fine/eval_batch_cli.py create --name demo --pairs 0,1,4,16
  python setup/coarse_to_fine/eval_batch_cli.py run <batch_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from setup.coarse_to_fine import annotations, deskew, eval_runs, masks, tre_eval
from setup.coarse_to_fine.field import (
    Candidate,
    Field,
    candidate_from_dict,
    fit_field,
    fit_gated,
    tau_for_keep,
)
from setup.coarse_to_fine.identity import pair_fingerprint
from setup.coarse_to_fine.reg_branches import cache_path, normalize_estimator, normalize_lam
from setup.coarse_to_fine.run import cache_candidates


def _emit(stage: str, **kv) -> None:
    parts = [f"stage={stage}"]
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


def _load_level_candidates(pair_id: int, level: int, lam: str) -> list[Candidate]:
    path = cache_path(pair_id, level, lam)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [candidate_from_dict(level, d) for d in payload.get("candidates", [])]


def _ensure_lam_caches(
    pair_id: int,
    levels: list[int],
    lam: str,
    force: bool,
) -> None:
    for level in sorted(levels):
        path = cache_path(pair_id, level, lam)
        if path.exists() and not force:
            _emit("cache_hit", pair=pair_id, lam=lam, level=level)
            continue
        _emit("cache_compute", pair=pair_id, lam=lam, level=level)
        cache_candidates(pair_id, level, levels, float("inf"), lam=lam)


def _fit_pair_lam_estimator(
    pair_id: int,
    levels: list[int],
    lam: str,
    estimator: str,
    wendland_eps: float,
    bspline_grid: int,
    bspline_reg: float,
) -> tuple[Field, dict]:
    import setup.coarse_to_fine.field as field_mod

    field_mod.WENDLAND_EPS = float(wendland_eps)
    field_mod.BSPLINE_GRID = int(bspline_grid)
    field_mod.BSPLINE_REG = float(bspline_reg)

    entries = annotations.load(pair_id)
    mask_entries = masks.load(pair_id)
    field = fit_field(
        annotations.to_anchors(entries, up_to_level=min(levels) - 1),
        field_estimator=estimator,
        wendland_epsilon=wendland_eps,
        bspline_grid=bspline_grid,
        bspline_reg=bspline_reg,
    )

    total_kept = 0
    total_seen = 0
    level_gates: dict[str, dict] = {}
    for level in sorted(levels):
        cands = _load_level_candidates(pair_id, level, lam)
        if not cands:
            continue
        masked = masks.masked_at(mask_entries, level, [c.tile_loc for c in cands])
        excluded = {
            e["tile_loc"]
            for e in entries
            if int(e["level"]) == level and e.get("type") == "exclude"
        }
        fit_cands = [
            c for c in cands if c.tile_loc not in masked and c.tile_loc not in excluded
        ]
        human = annotations.to_anchors(entries, up_to_level=level)
        keep = eval_runs.keep_for_level(level)
        if keep >= 1.0:
            tau = float("inf")
        else:
            tau = tau_for_keep(human, fit_cands, keep, field_estimator=estimator)
        field, kept = fit_gated(
            human, fit_cands, tau, field_estimator=estimator
        )
        level_gates[str(level)] = {
            "exclude_pct": 1.0 - keep,
            "keep": keep,
            "tau": None if tau == float("inf") else float(tau),
            "n_kept": len(kept),
            "n_seen": len(fit_cands),
        }
        total_kept += len(kept)
        total_seen += len(fit_cands)

    meta = {
        "lam": lam,
        "field_estimator": estimator,
        "exclude_pct_by_level": dict(eval_runs.EXCLUDE_PCT_BY_LEVEL),
        "level_gates": level_gates,
        "levels": levels,
        "n_kept": total_kept,
        "n_seen": total_seen,
        "n_human": len(entries),
        "wendland_eps": wendland_eps,
        "bspline_grid": bspline_grid,
        "bspline_reg": bspline_reg,
        "saved_depth": eval_runs.EVAL_DEPTH,
    }
    return field, meta


def _write_cell(
    batch_id: str,
    pair_id: int,
    lam: str,
    estimator: str,
    field: Field,
    meta: dict,
) -> dict:
    cell = eval_runs.cell_dir(batch_id, pair_id, lam, estimator)
    cell.mkdir(parents=True, exist_ok=True)

    # Full multi-depth JSON (overlay/TRE need depths["5"]); filename per plan.
    out_path = eval_runs.field_l5_path(batch_id, pair_id, lam, estimator)
    depths_out = {
        str(d): field.predict_tile_px(d) for d in range(eval_runs.EVAL_DEPTH + 1)
    }
    payload = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "fit_depth": eval_runs.EVAL_DEPTH,
        "depths": depths_out,
        **meta,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))

    deskew_store = deskew.load(pair_id)
    if deskew_store:
        (cell / "deskew.json").write_text(json.dumps(deskew_store, separators=(",", ":")))

    eval_runs.meta_path(batch_id, pair_id, lam, estimator).write_text(
        json.dumps(meta, indent=2)
    )

    points = tre_eval.load_landmarks(pair_id)
    w, h, scale = tre_eval.canvas_scale(pair_id)
    if points:
        errs = tre_eval.tre_field_file(points, out_path, w, h, scale)
        tre = tre_eval.annotate_tile_means(tre_eval.stats(errs), scale)
    else:
        tre = tre_eval.empty_err("no landmarks")
    tre.update({"lam": lam, "field_estimator": estimator, "n": len(points)})
    eval_runs.tre_path(batch_id, pair_id, lam, estimator).write_text(
        json.dumps(tre, indent=2)
    )
    return tre


def run_batch(batch_id: str) -> dict:
    manifest = eval_runs.read_manifest(batch_id)
    if manifest is None:
        raise FileNotFoundError(f"no batch {batch_id}")

    pairs = [int(p) for p in manifest["pairs"]]
    lams = [normalize_lam(x) for x in manifest.get("lams") or []]
    estimators = [normalize_estimator(x) for x in manifest.get("estimators") or []]
    cfg = {**eval_runs.default_config(), **(manifest.get("config") or {})}
    levels = sorted(int(x) for x in cfg["levels"])
    force = bool(cfg.get("force"))
    wendland_eps = float(cfg["wendland_eps"])
    bspline_grid = int(cfg["bspline_grid"])
    bspline_reg = float(cfg["bspline_reg"])

    jobs = [(p, lam, est) for p in pairs for lam in lams for est in estimators]
    total = len(jobs)
    done = 0
    eval_runs.write_status(
        batch_id,
        {
            "state": "running",
            "done": 0,
            "total": total,
            "detail": "start",
            "error": None,
            "started_at": int(time.time()),
        },
    )
    _emit("start", batch=batch_id, total=total)

    try:
        for pair_id, lam, estimator in jobs:
            done += 1
            detail = f"pair={pair_id} lam={lam} estimator={estimator}"
            _emit(
                "cell",
                pair=pair_id,
                lam=lam,
                estimator=estimator,
                done=done,
                total=total,
            )
            eval_runs.write_status(
                batch_id,
                {
                    "state": "running",
                    "done": done - 1,
                    "total": total,
                    "detail": detail,
                    "error": None,
                },
            )

            if (
                eval_runs.cell_complete(batch_id, pair_id, lam, estimator)
                and not force
            ):
                _emit("skip", pair=pair_id, lam=lam, estimator=estimator)
                eval_runs.write_status(
                    batch_id,
                    {
                        "state": "running",
                        "done": done,
                        "total": total,
                        "detail": detail,
                        "error": None,
                    },
                )
                continue

            _ensure_lam_caches(pair_id, levels, lam, force=force)
            field, meta = _fit_pair_lam_estimator(
                pair_id,
                levels,
                lam,
                estimator,
                wendland_eps,
                bspline_grid,
                bspline_reg,
            )
            _write_cell(batch_id, pair_id, lam, estimator, field, meta)
            eval_runs.write_status(
                batch_id,
                {
                    "state": "running",
                    "done": done,
                    "total": total,
                    "detail": detail,
                    "error": None,
                },
            )

        eval_runs.write_status(
            batch_id,
            {
                "state": "done",
                "done": total,
                "total": total,
                "detail": "complete",
                "error": None,
                "finished_at": int(time.time()),
            },
        )
        _emit("done", batch=batch_id, total=total)
        return {"ok": True, "batch_id": batch_id, "total": total}
    except Exception as e:
        eval_runs.write_status(
            batch_id,
            {
                "state": "error",
                "done": done,
                "total": total,
                "detail": "",
                "error": str(e),
                "finished_at": int(time.time()),
            },
        )
        _emit("error", batch=batch_id, msg=str(e).replace(" ", "_"))
        raise


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def cmd_list() -> None:
    _print_json({"batches": eval_runs.list_batches()})


def cmd_create(args: argparse.Namespace) -> None:
    pairs = [int(x) for x in args.pairs.split(",") if x.strip() != ""]
    config = eval_runs.default_config()
    if args.wendland_eps is not None:
        config["wendland_eps"] = float(args.wendland_eps)
    if args.bspline_grid is not None:
        config["bspline_grid"] = int(args.bspline_grid)
    if args.bspline_reg is not None:
        config["bspline_reg"] = float(args.bspline_reg)
    if args.force:
        config["force"] = True
    lams = [x.strip() for x in args.lams.split(",")] if args.lams else None
    estimators = (
        [x.strip() for x in args.estimators.split(",")] if args.estimators else None
    )
    try:
        man = eval_runs.create_batch(
            args.name,
            pairs,
            lams=lams,
            estimators=estimators,
            config=config,
            notes=args.notes,
            batch_id=args.id,
        )
    except FileExistsError as e:
        _print_json({"ok": False, "error": str(e)})
        sys.exit(1)
    _print_json({"ok": True, "manifest": man})


def cmd_run(args: argparse.Namespace) -> None:
    _print_json(run_batch(args.batch_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--pairs", required=True, help="comma-separated pair ids")
    c.add_argument("--id", default=None, help="optional batch id (slug)")
    c.add_argument("--lams", default=None, help="comma-separated, default all")
    c.add_argument("--estimators", default=None, help="comma-separated, default all")
    c.add_argument("--wendland-eps", type=float, default=None)
    c.add_argument("--bspline-grid", type=int, default=None)
    c.add_argument("--bspline-reg", type=float, default=None)
    c.add_argument("--force", action="store_true")
    c.add_argument("--notes", default="")

    r = sub.add_parser("run")
    r.add_argument("batch_id")

    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "create":
        cmd_create(args)
    elif args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
