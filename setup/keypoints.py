"""
SuperPoint keypoint detection on HE tiles.

Runs SuperPoint on each tile's he.png (already at CNN input size 344×512)
and writes detected keypoints alongside their confidence scores to a per-tile
keypoints.json.  No adaptive threshold correction — a single pass at a fixed
conf_thresh.  The full confidence distribution is preserved so downstream
training code can apply its own threshold.

Usage — batch (next N unprocessed pairs, all depths):
    python keypoints.py --pairs N [--conf-thresh 0.015] [--nms-dist 12] [--force]

Usage — single depth:
    python keypoints.py <pair_id> <depth> [--conf-thresh 0.015] [--nms-dist 12] [--force]

A pair is "unprocessed" when it contains no keypoints.json files.

Output per tile:
    data/cropped/<pair>/d<depth>/<tile>/keypoints.json
    {
      "keypoints": [[x, y, conf], ...],
      "keypoint_count": 87,
      "conf_thresh": 0.015,
      "nms_dist": 12
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "explore" / "pretraining_compare_detectors"))
sys.path.insert(0, str(ROOT / "external"))

from SuperPointPretrainedNetwork import SuperPointFrontend  # noqa: E402

DATA_ROOT    = ROOT / "data" / "cropped"
WEIGHTS      = ROOT / "external" / "superpoint_v1.pth"
KEYPTS_FILE  = "keypoints.json"

DEFAULT_CONF_THRESH = 0.015
DEFAULT_NMS_DIST    = 12


# ── Detection ────────────────────────────────────────────────────────────────

def load_frontend(conf_thresh: float, nms_dist: int) -> SuperPointFrontend:
    cuda = torch.cuda.is_available()
    return SuperPointFrontend(
        weights_path=str(WEIGHTS),
        nms_dist=nms_dist,
        conf_thresh=conf_thresh,
        nn_thresh=0.7,
        cuda=cuda,
    )


def detect(frontend: SuperPointFrontend, he_path: Path) -> list[list[float]]:
    """Run SuperPoint on he.png; return [[x, y, conf], ...]."""
    img = cv2.imread(str(he_path), cv2.IMREAD_GRAYSCALE)
    gray = img.astype(np.float32) / 255.0
    pts, _, _ = frontend.run(gray)
    if pts.ndim != 2 or pts.shape[1] == 0:
        return []
    return [[float(pts[0, i]), float(pts[1, i]), float(pts[2, i])]
            for i in range(pts.shape[1])]


# ── Depth / pair processing ───────────────────────────────────────────────────

def process_depth(
    depth_dir: Path,
    frontend: SuperPointFrontend,
    conf_thresh: float,
    nms_dist: int,
    force: bool,
) -> tuple[int, int]:
    done = skipped = 0
    for tile_dir in sorted(d for d in depth_dir.iterdir() if d.is_dir()):
        out_file = tile_dir / KEYPTS_FILE
        if out_file.exists() and not force:
            skipped += 1
            continue
        he_path = tile_dir / "he.png"
        if not he_path.exists():
            skipped += 1
            continue
        try:
            keypoints = detect(frontend, he_path)
        except Exception as exc:
            print(f"    ERROR {tile_dir.name}: {exc}")
            skipped += 1
            continue
        out_file.write_text(json.dumps({
            "keypoints":      keypoints,
            "keypoint_count": len(keypoints),
            "conf_thresh":    conf_thresh,
            "nms_dist":       nms_dist,
        }))
        done += 1
    return done, skipped


def pair_has_keypoints(pair_dir: Path) -> bool:
    return any(pair_dir.rglob(KEYPTS_FILE))


def process(pair_id: int, depth: int, frontend: SuperPointFrontend,
            conf_thresh: float, nms_dist: int, force: bool) -> None:
    depth_dir = DATA_ROOT / str(pair_id) / f"d{depth}"
    if not depth_dir.is_dir():
        sys.exit(f"Directory not found: {depth_dir}")
    done, skip = process_depth(depth_dir, frontend, conf_thresh, nms_dist, force)
    print(f"pair {pair_id}  level {depth}  done={done}  skipped={skip}")


def process_pairs(n: int, frontend: SuperPointFrontend,
                  conf_thresh: float, nms_dist: int, force: bool) -> None:
    if not DATA_ROOT.is_dir():
        sys.exit(f"Data root not found: {DATA_ROOT}")

    pair_dirs = sorted(
        (d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    queued = [d for d in pair_dirs if force or not pair_has_keypoints(d)]
    batch  = queued[:n]

    if not batch:
        print("Nothing to process — all pairs already have keypoints (use --force to rerun).")
        return

    total_done = total_skip = 0
    for pair_dir in batch:
        pair_id   = pair_dir.name
        depth_dirs = sorted(
            (d for d in pair_dir.iterdir() if d.is_dir() and d.name.startswith("d")),
            key=lambda d: int(d.name[1:]),
        )
        for depth_dir in depth_dirs:
            level      = int(depth_dir.name[1:])
            tile_count = sum(1 for d in depth_dir.iterdir() if d.is_dir())
            print(f"pair {pair_id}  level {level}  ({tile_count} tiles) ...", end="", flush=True)
            done, skip = process_depth(depth_dir, frontend, conf_thresh, nms_dist, force)
            print(f"  {done} computed, {skip} skipped")
            total_done += done
            total_skip += skip

    print(f"\ntotal: {total_done} computed, {total_skip} skipped")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    argv  = sys.argv[1:]
    force = "--force" in argv or "-f" in argv
    argv  = [a for a in argv if a not in ("--force", "-f")]

    conf_thresh = DEFAULT_CONF_THRESH
    nms_dist    = DEFAULT_NMS_DIST

    for flag, attr in (("--conf-thresh", "conf_thresh"), ("--nms-dist", "nms_dist")):
        if flag in argv:
            idx  = argv.index(flag)
            if idx + 1 >= len(argv):
                sys.exit(f"{flag} requires a value")
            val  = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]
            if attr == "conf_thresh":
                conf_thresh = float(val)
            else:
                nms_dist = int(val)

    print(f"Loading SuperPoint  (conf_thresh={conf_thresh}  nms_dist={nms_dist}  "
          f"cuda={torch.cuda.is_available()}) …", flush=True)
    frontend = load_frontend(conf_thresh, nms_dist)
    print("Model loaded.\n", flush=True)

    if "--pairs" in argv:
        idx = argv.index("--pairs")
        if idx + 1 >= len(argv):
            sys.exit("--pairs requires a number argument")
        n = int(argv[idx + 1])
        process_pairs(n, frontend, conf_thresh, nms_dist, force=force)
        return

    if len(argv) < 2:
        sys.exit(
            "Usage:\n"
            "  python keypoints.py <pair_id> <depth> [--conf-thresh F] [--nms-dist N] [--force]\n"
            "  python keypoints.py --pairs N  [--conf-thresh F] [--nms-dist N] [--force]"
        )
    process(int(argv[0]), int(argv[1]), frontend, conf_thresh, nms_dist, force=force)


if __name__ == "__main__":
    main()
