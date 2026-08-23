"""Are IHC crops at HE-derived tile coords actually on tissue?

Crops the same tile indices from the HE page and the raw IHC page and reports
mean intensity plus SuperPoint keypoint yield. Background is ~255 (white).
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "setup" / "live_crop"), str(REPO / "introducing_superpoint")]

from setup import datasets as ds
from setup.coarse_to_fine import sp_rot_train as store
from setup.coarse_to_fine import sp_rot_train_data as data

ds.set_active_dataset("muromi")
import crop_core
from training import build_model

DEPTH = 5
LEVEL = 2
N_TILES = 40

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(
    store.DEFAULT_CONFIG["init_weights"],
    device=dev,
    nms_radius=int(store.DEFAULT_CONFIG["sp_nms_dist"]),
    detection_threshold=float(store.DEFAULT_CONFIG["gt_conf_thresh"]),
    max_num_keypoints=store.DEFAULT_CONFIG.get("gt_max_kpts"),
)

print(f"depth={DEPTH} level={LEVEL} sampling {N_TILES} masked-in tiles per pair")
print(f"{'pair':>5} {'side':>5} {'mean_int':>9} {'blank%':>7} {'med_kp':>7} {'kp<50%':>7}")
for pid in range(0, 6):
    try:
        tiles = data.scan_tiles([pid], DEPTH)
    except Exception as e:
        print(f"{pid:>5}  scan failed: {e}")
        continue
    if not tiles:
        print(f"{pid:>5}  no tiles")
        continue
    step = max(1, len(tiles) // N_TILES)
    sample = tiles[::step][:N_TILES]
    for side in ("he", "ihc"):
        page = crop_core.whole_gray(pid, side, LEVEL)
        if page is None:
            print(f"{pid:>5} {side:>5}  no page")
            continue
        means, kps = [], []
        for t in sample:
            base, _w, _v, _H = data.make_warp_pair(
                page, t, depth=DEPTH, preview_level=LEVEL,
                src_size=768, out_size=512, theta_deg=0.0,
            )
            means.append(float(base.mean()))
            x = torch.from_numpy(base.astype(np.float32) / 255.0)[None, None].to(dev)
            gt = store.detect_pseudo_gt(model, x, dev)
            kps.append(int(gt[0].shape[0]))
        means, kps = np.array(means), np.array(kps)
        print(
            f"{pid:>5} {side:>5} {means.mean():>9.1f} "
            f"{100.0 * float((means > 245).mean()):>6.1f}% "
            f"{int(np.median(kps)):>7} {100.0 * float((kps < 50).mean()):>6.1f}%"
        )
