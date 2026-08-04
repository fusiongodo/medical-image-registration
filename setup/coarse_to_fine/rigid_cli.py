"""
CLI for SuperPoint + LightGlue rigid lab (light_v1).

Usage (JSON on stdout; run prints stage=* progress lines then a final result):
    rigid_cli.py run   <pair> [--level N] [--pre-rot DEG] [--hyperparams JSON]
    rigid_cli.py get   <pair>
    rigid_cli.py run-result <pair>
    rigid_cli.py progress <pair>
    rigid_cli.py save  <pair>
    rigid_cli.py clear <pair> [--run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from setup.coarse_to_fine import rigid_sp_lg as rigid


def cmd_run(args: argparse.Namespace) -> dict:
    hp = json.loads(args.hyperparams) if args.hyperparams else {}
    try:
        return rigid.run(
            args.pair,
            preview_level=args.level,
            pre_rotation_deg=args.pre_rot,
            hyperparams=hp,
        )
    except Exception as exc:
        rigid._progress(args.pair, "error", str(exc))
        return {"error": str(exc)}


def cmd_get(args: argparse.Namespace) -> dict:
    saved = rigid.load(args.pair)
    run = rigid.load_run(args.pair)
    matches = rigid.load_matches(args.pair)
    return {
        "saved": saved,
        "has_run": run is not None,
        "run": run,
        "matches": matches,
        "has_matches": matches is not None,
    }


def cmd_run_result(args: argparse.Namespace) -> dict:
    run = rigid.load_run(args.pair)
    if run is None:
        return {"error": "no run result"}
    return run


def cmd_progress(args: argparse.Namespace) -> dict:
    return rigid.load_progress(args.pair) or {"stage": None}


def cmd_save(args: argparse.Namespace) -> dict:
    return rigid.save_from_run(args.pair)


def cmd_clear(args: argparse.Namespace) -> dict:
    rigid.clear(args.pair)
    cleared_run = False
    if args.run:
        pair_dir = rigid.RIGID_ROOT / str(args.pair)
        if pair_dir.exists():
            shutil.rmtree(pair_dir)
            cleared_run = True
    caches = rigid.clear_caches(args.pair)
    return {"ok": True, "cleared": args.pair, "cleared_run": cleared_run, "caches_cleared": caches}


def cmd_field_fit(args: argparse.Namespace) -> dict:
    try:
        return rigid.field_fit(
            args.pair,
            field_estimator=args.estimator,
            wendland_epsilon=args.wendland_eps,
            bspline_grid=args.bspline_grid,
            bspline_reg=args.bspline_reg,
            inliers_only=not args.all_matches,
            mode=args.mode,
        )
    except Exception as exc:
        return {"error": str(exc)}


def cmd_reclassify(args: argparse.Namespace) -> dict:
    try:
        return rigid.reclassify_inliers(args.pair, float(args.inlier_px))
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("pair", type=int)
    p_run.add_argument("--level", type=int, default=rigid.DEFAULT_PREVIEW_LEVEL)
    p_run.add_argument("--pre-rot", type=float, default=0.0)
    p_run.add_argument("--hyperparams", type=str, default="")
    p_run.set_defaults(func=cmd_run)

    p_get = sub.add_parser("get")
    p_get.add_argument("pair", type=int)
    p_get.set_defaults(func=cmd_get)

    p_rr = sub.add_parser("run-result")
    p_rr.add_argument("pair", type=int)
    p_rr.set_defaults(func=cmd_run_result)

    p_pr = sub.add_parser("progress")
    p_pr.add_argument("pair", type=int)
    p_pr.set_defaults(func=cmd_progress)

    p_save = sub.add_parser("save")
    p_save.add_argument("pair", type=int)
    p_save.set_defaults(func=cmd_save)

    p_clear = sub.add_parser("clear")
    p_clear.add_argument("pair", type=int)
    p_clear.add_argument("--run", action="store_true")
    p_clear.set_defaults(func=cmd_clear)

    p_ff = sub.add_parser("field-fit")
    p_ff.add_argument("pair", type=int)
    p_ff.add_argument("--estimator", default="tps", choices=("tps", "wendland", "bspline"))
    p_ff.add_argument("--wendland-eps", type=float, default=None)
    p_ff.add_argument("--bspline-grid", type=int, default=None)
    p_ff.add_argument("--bspline-reg", type=float, default=None)
    p_ff.add_argument("--all-matches", action="store_true")
    p_ff.add_argument(
        "--mode",
        default="residual_after_rigid",
        choices=("residual_after_rigid", "direct"),
    )
    p_ff.set_defaults(func=cmd_field_fit)

    p_rc = sub.add_parser("reclassify")
    p_rc.add_argument("pair", type=int)
    p_rc.add_argument("--inlier-px", type=float, required=True)
    p_rc.set_defaults(func=cmd_reclassify)

    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
