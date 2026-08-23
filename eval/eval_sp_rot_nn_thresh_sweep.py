"""
Sweep Magicleap two-way NN thresholds on frozen overfit weights.

  python eval/eval_sp_rot_nn_thresh_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "setup" / "live_crop"))
sys.path.insert(0, str(REPO / "introducing_superpoint"))

from setup import datasets
from setup.coarse_to_fine import sp_rot_train as store
from setup.coarse_to_fine import sp_rot_train_eval as beval
from training import DEFAULT_WEIGHTS

ROOT = REPO / "data" / "sp_rot_train" / "_overfit_original_loss"
TILE = {"pair_id": 0, "x": 12, "y": 10, "loc": "12_10"}
ANGLES = [0.0, 120.0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
WEIGHTS = {
    "magicleap": Path(DEFAULT_WEIGHTS),
    "a2_ce_3k": ROOT / "overfit_product_t1_a2_b2_d0_p0_12_10_a2_ce.pt",
    "a2_ce_33k": ROOT / "overfit_product_t1_a2_b2_d0_p0_12_10_a2_ce_s3001_33000.pt",
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compact(part: dict) -> dict:
    return {
        "n_pass": part.get("n_pass"),
        "n_total": part.get("n_total"),
        "pass_rate": part.get("pass_rate"),
        "cells": [
            {
                "angle": c.get("angle"),
                "auto_pass": c.get("auto_pass"),
                "rot_err_deg": c.get("rot_err_deg"),
                "trans_err_rel": c.get("trans_err_rel"),
                "n_matches": c.get("n_matches"),
                "n_inliers": c.get("n_inliers"),
                "n_kp0": c.get("n_kp0"),
                "n_kp1": c.get("n_kp1"),
                "error": c.get("error"),
            }
            for c in part.get("cells") or []
        ],
    }


def main() -> None:
    datasets.set_active_dataset("muromi")
    cfg = {**store.DEFAULT_CONFIG}
    device = _device()
    print("device", device)

    out = {
        "tile": TILE,
        "angles": ANGLES,
        "thresholds": THRESHOLDS,
        "gate": {
            "max_rot_err_deg": beval.MAX_ROT_ERR_DEG,
            "max_trans_err_rel": beval.MAX_TRANS_ERR_REL,
        },
        "sp_nms_dist": int(cfg["sp_nms_dist"]),
        "runs": {},
    }

    for name, path in WEIGHTS.items():
        if not path.is_file():
            print("skip missing", path)
            continue
        model = beval.load_sp_model(path, device, nms_radius=int(cfg["sp_nms_dist"]))
        by_thresh = {}
        for thr in THRESHOLDS:
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
                nn_thresh=float(thr),
            )
            nn = _compact(ev["nn"])
            by_thresh[str(thr)] = nn
            cells = {float(c["angle"]): c for c in nn["cells"]}
            c0, c120 = cells.get(0.0, {}), cells.get(120.0, {})
            print(
                f"{name} thr={thr:.1f} "
                f"k={nn['n_pass']}/{nn['n_total']} "
                f"0° m={c0.get('n_matches')} inl={c0.get('n_inliers')} pass={c0.get('auto_pass')} "
                f"120° m={c120.get('n_matches')} inl={c120.get('n_inliers')} "
                f"rot={c120.get('rot_err_deg')} pass={c120.get('auto_pass')}"
            )
        out["runs"][name] = {"weights": str(path), "by_thresh": by_thresh}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    path = ROOT / "nn_thresh_sweep.json"
    ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
