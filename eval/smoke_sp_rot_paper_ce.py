"""
Local smoke test for the original-loss overfit path: paper_ce train step and
the dual-matcher eval. No run config needed.

  python eval/smoke_sp_rot_paper_ce.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "setup" / "live_crop"))
sys.path.insert(0, str(REPO / "introducing_superpoint"))

from setup import datasets
from setup.coarse_to_fine import sp_rot_train as store
from setup.coarse_to_fine import sp_rot_train_data as data
from setup.coarse_to_fine import sp_rot_train_eval as beval
import crop_core

TILE = {"pair_id": 0, "x": 12, "y": 10, "loc": "12_10"}
ANGLES = [0.0, 120.0]


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    datasets.set_active_dataset("muromi")
    cfg = {**store.DEFAULT_CONFIG, "kp_loss": "paper_ce", "desc_max_cells": 0}
    device = _device()
    page = crop_core.whole_gray(0, "he", int(cfg["preview_level"]))
    if page is None:
        raise SystemExit("no L2 HE page")

    samples = []
    for theta in ANGLES:
        base, warped, valid, H = data.make_warp_pair(
            page,
            TILE,
            depth=int(cfg["depth"]),
            preview_level=int(cfg["preview_level"]),
            src_size=int(cfg["src_size"]),
            out_size=int(cfg["out_size"]),
            theta_deg=theta,
        )
        samples.append(
            {
                "image": torch.from_numpy(base.astype(np.float32) / 255.0).unsqueeze(0),
                "warped": torch.from_numpy(warped.astype(np.float32) / 255.0).unsqueeze(0),
                "valid_mask": torch.from_numpy((valid > 127).astype(np.float32)),
                "homography": torch.from_numpy(H),
                "theta_deg": theta,
                "pair_id": 0,
                "side": "he",
            }
        )

    from training import build_model

    model = build_model(
        cfg.get("init_weights"),
        device=device,
        nms_radius=int(cfg["sp_nms_dist"]),
        detection_threshold=float(cfg["gt_conf_thresh"]),
        max_num_keypoints=cfg.get("gt_max_kpts"),
    )
    with torch.no_grad():
        for s in samples:
            img = s["image"].unsqueeze(0).to(device)
            Ht = s["homography"].unsqueeze(0).to(device)
            pseudo = store.detect_pseudo_gt(model, img, device)
            s["gt_base"] = [
                g.detach().cpu() for g in store.filter_gt_in_frame(pseudo, int(cfg["out_size"]))
            ]
            s["gt_warp"] = [
                g.detach().cpu()
                for g in store.filter_gt_in_frame(
                    store.warp_gt_points(pseudo, Ht), int(cfg["out_size"])
                )
            ]
    print("n_gt base", [int(s["gt_base"][0].shape[0]) for s in samples])
    print("n_gt warp", [int(s["gt_warp"][0].shape[0]) for s in samples])

    batch = {
        "image": torch.stack([s["image"] for s in samples], dim=0),
        "warped": torch.stack([s["warped"] for s in samples], dim=0),
        "valid_mask": torch.stack([s["valid_mask"] for s in samples], dim=0),
        "homography": torch.stack([s["homography"] for s in samples], dim=0),
        "theta_deg": [s["theta_deg"] for s in samples],
        "pair_id": [0, 0],
        "side": ["he", "he"],
    }
    gt_base = [s["gt_base"][0].to(device) for s in samples]
    gt_warp = [s["gt_warp"][0].to(device) for s in samples]

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    for step in (1, 2):
        m = store.train_step(model, batch, opt, device, cfg, gt_base=gt_base, gt_warp=gt_warp)
        print(
            f"step {step} kp={m['loss_kp']:.6f} fn={m['loss_fn']:.6f} "
            f"fp={m['loss_fp']:.6f} desc={m['loss_desc']:.6f} total={m['loss_total']:.6f} "
            f"n_pos={m['n_pos']} n_neg={m['n_neg']}"
        )

    ev = beval.evaluate_tile_matchers(
        model,
        [TILE],
        angles=ANGLES,
        device=device,
        extract_resize=int(cfg["extract_resize"]),
        nms=int(cfg["sp_nms_dist"]),
        depth=int(cfg["depth"]),
        preview_level=int(cfg["preview_level"]),
        src_size=int(cfg["src_size"]),
        out_size=int(cfg["out_size"]),
    )
    for kind in ("nn", "lg"):
        part = ev[kind]
        print(f"{kind} k={part['n_pass']}/{part['n_total']}")
        for c in part["cells"]:
            print(
                f"  th={c['angle']:g} n_matches={c['n_matches']} "
                f"n_inliers={c['n_inliers']} pass={c['auto_pass']} "
                f"kp={c['n_kp0']}/{c['n_kp1']} err={c['error']}"
            )


if __name__ == "__main__":
    main()
