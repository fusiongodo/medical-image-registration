"""
Re-score Magicleap + 12_10 overfit weights with two-way NN (no LightGlue).

  python eval/eval_sp_rot_overfit_nn.py
"""

from __future__ import annotations

import json
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
FIG = ROOT / "figures"
TILE = {"pair_id": 0, "x": 12, "y": 10, "loc": "12_10"}
ALL_ANGLES = list(range(0, 360, 30))
OVERLAY_ANGLES = [0, 120]
NN_THRESH = 0.7
RUNS = {
    "a12": {
        "weights": ROOT / "overfit_product_t1_a12_b12_d0_p0_12_10.pt",
        "angles": ALL_ANGLES,
        "lg_k_over_n": "1/12",
    },
    "a3": {
        "weights": ROOT / "overfit_product_t1_a3_b3_d0_p0_12_10_a3.pt",
        "angles": [0, 120, 240],
        "lg_k_over_n": "1/3",
    },
    "a2": {
        "weights": ROOT / "overfit_product_t1_a2_b2_d0_p0_12_10_a2.pt",
        "angles": [0, 120],
        "lg_k_over_n": "1/2",
    },
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compact(ev: dict) -> dict:
    cells = ev.get("cells") or []
    return {
        "n_pass": ev.get("n_pass"),
        "n_total": ev.get("n_total"),
        "pass_rate": ev.get("pass_rate"),
        "n_error": ev.get("n_error"),
        "by_angle": ev.get("by_angle"),
        "cells": [
            {
                "angle": c.get("angle"),
                "auto_pass": c.get("auto_pass"),
                "rot_err_deg": c.get("rot_err_deg"),
                "trans_err_rel": c.get("trans_err_rel"),
                "n_matches": c.get("n_matches"),
                "n_inliers": c.get("n_inliers"),
                "error": c.get("error"),
            }
            for c in cells
        ],
    }


def _subset(ev: dict, angles: list[float]) -> dict:
    want = {float(a) for a in angles}
    cells = [c for c in ev.get("cells") or [] if float(c.get("angle")) in want]
    n_pass = sum(1 for c in cells if c.get("auto_pass"))
    n_err = sum(1 for c in cells if c.get("error"))
    by_angle = {}
    for a in angles:
        key = str(a)
        src = (ev.get("by_angle") or {}).get(key) or {}
        by_angle[key] = src
    n_total = len(cells)
    return {
        "n_pass": n_pass,
        "n_total": n_total,
        "pass_rate": (n_pass / n_total) if n_total else None,
        "n_error": n_err,
        "by_angle": by_angle,
        "cells": cells,
    }


def _overlay(model, device, page, cfg, theta: float) -> np.ndarray:
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
    pts0, pts1, _sc = beval.match_nn_two_way(f0, f1, NN_THRESH)
    inliers = None
    if len(pts0) >= 2:
        _R, _t, inliers, _st = rigid_sp_lg.fit_rigid_kabsch(
            pts0, pts1, float(rigid_sp_lg.DEFAULT_HYPERPARAMS.get("rigid_inlier_px", 3.0))
        )
    return rigid_sp_lg._draw_matches(base, warped, pts0, pts1, inliers)


def _save(name: str, img: np.ndarray) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    cv2.imwrite(str(path), img)
    print("wrote", path)


def _eval_model(model, device, cfg, angles: list[float]) -> dict:
    return beval.evaluate_tiles(
        None,
        [TILE],
        angles=angles,
        extract_resize=int(cfg["extract_resize"]),
        nms=int(cfg["sp_nms_dist"]),
        depth=int(cfg["depth"]),
        preview_level=int(cfg["preview_level"]),
        src_size=int(cfg["src_size"]),
        out_size=int(cfg["out_size"]),
        max_tiles=None,
        dataset=str(cfg.get("dataset") or "muromi"),
        model=model,
        device=device,
        match_kind="nn",
        nn_thresh=NN_THRESH,
    )


def main() -> None:
    datasets.set_active_dataset("muromi")
    cfg = {**store.DEFAULT_CONFIG}
    page = crop_core.whole_gray(0, "he", int(cfg["preview_level"]))
    if page is None:
        raise SystemExit("no L2 HE page")
    device = _device()
    print("device", device)

    before = beval.load_sp_model(DEFAULT_WEIGHTS, device, nms_radius=int(cfg["sp_nms_dist"]))
    before_full = _compact(_eval_model(before, device, cfg, ALL_ANGLES))
    print(
        "nn magicleap",
        f"{before_full['n_pass']}/{before_full['n_total']}",
        "ok",
        sorted(float(c["angle"]) for c in before_full["cells"] if c.get("auto_pass")),
    )
    for th in OVERLAY_ANGLES:
        _save(f"matches_nn_before_th{int(th)}.png", _overlay(before, device, page, cfg, th))
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()

    runs = {}
    for name, spec in RUNS.items():
        model = beval.load_sp_model(spec["weights"], device, nms_radius=int(cfg["sp_nms_dist"]))
        after = _compact(_eval_model(model, device, cfg, spec["angles"]))
        base = _subset(before_full, spec["angles"])
        print(
            f"nn {name} lg={spec['lg_k_over_n']} "
            f"nn_before={base['n_pass']}/{base['n_total']} "
            f"nn_after={after['n_pass']}/{after['n_total']} "
            f"ok={sorted(float(c['angle']) for c in after['cells'] if c.get('auto_pass'))}"
        )
        runs[name] = {
            "lg_k_over_n": spec["lg_k_over_n"],
            "nn_before": base,
            "nn_after": after,
        }
        for th in OVERLAY_ANGLES:
            if float(th) in {float(a) for a in spec["angles"]}:
                _save(
                    f"matches_nn_{name}_after_th{int(th)}.png",
                    _overlay(model, device, page, cfg, th),
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {
        "tile": TILE,
        "match_kind": "nn",
        "nn_thresh": NN_THRESH,
        "gate": {
            "max_rot_err_deg": beval.MAX_ROT_ERR_DEG,
            "max_trans_err_rel": beval.MAX_TRANS_ERR_REL,
        },
        "sp_nms_dist": int(cfg["sp_nms_dist"]),
        "magicleap": before_full,
        "runs": runs,
    }
    path = ROOT / "nn_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
