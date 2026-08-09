"""Pre-generate panel thumbs for an SP-rot run (one PIL process, fast)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from setup.coarse_to_fine import sp_rot_bench as bench

NAMES = ("he.png", "ihc_rigid.png", "ihc_prerot.png")


def warm_run(run_id: str, widths: list[int]) -> dict:
    root = bench.run_dir(run_id)
    if not root.is_dir():
        raise FileNotFoundError(run_id)
    made = 0
    skipped = 0
    for pair_dir in sorted(root.iterdir()):
        if not pair_dir.is_dir() or not pair_dir.name.isdigit():
            continue
        for ang_dir in sorted(pair_dir.iterdir()):
            if not ang_dir.is_dir():
                continue
            thumb_dir = ang_dir / ".thumbs"
            for name in NAMES:
                src = ang_dir / name
                if not src.is_file():
                    continue
                with Image.open(src) as im0:
                    for w in widths:
                        dest = thumb_dir / f"{name}.w{w}.png"
                        if dest.is_file():
                            skipped += 1
                            continue
                        thumb_dir.mkdir(parents=True, exist_ok=True)
                        im = im0.copy()
                        im.thumbnail((w, w), Image.Resampling.BILINEAR)
                        im.save(dest, format="PNG", optimize=True)
                        made += 1
    return {"run_id": run_id, "made": made, "skipped": skipped, "widths": widths}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_id", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--widths", default="720,960")
    args = p.parse_args()
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    run_ids: list[str] = []
    if args.all:
        if bench.SP_ROT_ROOT.is_dir():
            run_ids = [d.name for d in bench.SP_ROT_ROOT.iterdir() if (d / "manifest.json").is_file()]
    elif args.run_id:
        run_ids = [args.run_id]
    else:
        raise SystemExit("pass run_id or --all")
    for rid in run_ids:
        out = warm_run(rid, widths)
        print(out, flush=True)


if __name__ == "__main__":
    main()
