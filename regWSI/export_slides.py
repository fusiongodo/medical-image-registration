"""
Export RGB whole-slide inputs for DeeperHistReg at the level-5 canvas.

Default SCALE=1 → (32*512) x (32*344) = 16384 x 11008.

Usage:
  python regWSI/export_slides.py <pair_id>
  python regWSI/export_slides.py --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))

import crop_core

from regWSI import paths

_BUILD = Path(__file__).resolve().with_name("build_rgb_page.py")


def _export_side(image_id: int, page_idx: int, out: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(_BUILD),
            str(image_id),
            str(page_idx),
            str(paths.GRID),
            str(paths.TILE_W),
            str(paths.TILE_H),
            str(out),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def export_pair(pair_id: int, force: bool = False) -> dict:
    chosen = crop_core.choose_page(pair_id, paths.LEVEL)
    if chosen is None:
        raise RuntimeError(f"no pyramid page for pair {pair_id} level {paths.LEVEL}")
    page_idx = chosen[0]
    he_id, ihc_id = crop_core.pair_image_ids(pair_id)
    paths.ensure_pair_dirs(pair_id)
    he_out = paths.he_tiff(pair_id)
    ihc_out = paths.ihc_tiff(pair_id)
    if force or not he_out.is_file():
        _export_side(he_id, page_idx, he_out)
    if force or not ihc_out.is_file():
        _export_side(ihc_id, page_idx, ihc_out)
    return {
        "pair_id": pair_id,
        "page": page_idx,
        "he_id": he_id,
        "ihc_id": ihc_id,
        "he": str(he_out),
        "ihc": str(ihc_out),
        "canvas": [paths.CANVAS_W, paths.CANVAS_H],
        "scale": paths.SCALE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair", nargs="?", type=int, help="pair index")
    ap.add_argument("--all", action="store_true", help="export every pair in labels")
    ap.add_argument("--force", action="store_true", help="overwrite existing tiffs")
    args = ap.parse_args()
    if args.all:
        n = crop_core.num_pairs()
        for i in range(n):
            print(f"export pair {i}/{n - 1} …", flush=True)
            print(export_pair(i, force=args.force), flush=True)
        return
    if args.pair is None:
        ap.error("pair or --all required")
    print(export_pair(args.pair, force=args.force))


if __name__ == "__main__":
    main()
