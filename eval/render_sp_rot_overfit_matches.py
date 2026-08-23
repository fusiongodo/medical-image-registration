"""
Render SuperPoint+LightGlue match overlays for the 12_10 overfit tile.

  python eval/render_sp_rot_overfit_matches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "setup" / "live_crop"))
sys.path.insert(0, str(REPO / "introducing_superpoint"))

from setup import datasets
from setup.coarse_to_fine import rigid_sp_lg
from setup.coarse_to_fine import sp_rot_train as store
from setup.coarse_to_fine import sp_rot_train_data as data
from setup.coarse_to_fine import sp_rot_train_eval as beval
import crop_core
from training import DEFAULT_WEIGHTS

ROOT = REPO / "data" / "sp_rot_train" / "_overfit_nms"
OUT = ROOT / "figures"
TILE = {"pair_id": 0, "x": 12, "y": 10, "loc": "12_10"}
ALL_ANGLES = list(range(0, 360, 30))
RUNS = {
    "a12": {
        "weights": ROOT / "overfit_product_t1_a12_b12_d0_p0_12_10.pt",
        "angles": ALL_ANGLES,
    },
    "a3": {
        "weights": ROOT / "overfit_product_t1_a3_b3_d0_p0_12_10_a3.pt",
        "angles": [0, 120, 240],
    },
    "a2": {
        "weights": ROOT / "overfit_product_t1_a2_b2_d0_p0_12_10_a2.pt",
        "angles": [0, 120],
    },
}


def _one(model, matcher, device, page, cfg, theta: float) -> np.ndarray:
    base, warped, _valid, _H = data.make_warp_pair(
        page,
        TILE,
        depth=int(cfg["depth"]),
        preview_level=int(cfg["preview_level"]),
        src_size=int(cfg["src_size"]),
        out_size=int(cfg["out_size"]),
        theta_deg=float(theta),
    )
    f0, _ = beval.extract_feats(model, base, device, int(cfg["extract_resize"]))
    f1, _ = beval.extract_feats(model, warped, device, int(cfg["extract_resize"]))
    hp = {**rigid_sp_lg.DEFAULT_HYPERPARAMS, "sp_nms_dist": int(cfg["sp_nms_dist"])}
    pts0, pts1, _sc = beval.match_lg(f0, f1, device, hp, matcher=matcher)
    inliers = None
    if len(pts0) >= 2:
        _R, _t, inliers, _st = rigid_sp_lg.fit_rigid_kabsch(
            pts0, pts1, float(hp.get("rigid_inlier_px", 3.0))
        )
    return rigid_sp_lg._draw_matches(base, warped, pts0, pts1, inliers)


def _save(name: str, img: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    cv2.imwrite(str(path), img)
    print("wrote", path)


def main() -> None:
    datasets.set_active_dataset("muromi")
    cfg = store.load_config("first")
    page = crop_core.whole_gray(0, "he", int(cfg["preview_level"]))
    if page is None:
        raise SystemExit("no L2 HE page")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = beval.build_lg_matcher(device, rigid_sp_lg.DEFAULT_HYPERPARAMS)

    before = beval.load_sp_model(DEFAULT_WEIGHTS, device, nms_radius=int(cfg["sp_nms_dist"]))
    for th in ALL_ANGLES:
        _save(f"matches_before_th{int(th)}.png", _one(before, matcher, device, page, cfg, th))
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for run, spec in RUNS.items():
        model = beval.load_sp_model(spec["weights"], device, nms_radius=int(cfg["sp_nms_dist"]))
        for th in spec["angles"]:
            _save(
                f"matches_{run}_after_th{int(th)}.png",
                _one(model, matcher, device, page, cfg, th),
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
