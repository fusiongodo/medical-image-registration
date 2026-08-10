"""B1 rigid eval using introducing_superpoint weights + LightGlue matcher."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))
sys.path.insert(0, str(REPO_ROOT / "introducing_superpoint"))

import conf
from setup.coarse_to_fine import lam_sp_lg
from setup.coarse_to_fine import rigid_sp_lg
from setup.coarse_to_fine import sp_rot_bench as bench

MAX_ROT_ERR_DEG = 1.0
MAX_TRANS_ERR_REL = 0.055
DEFAULT_EXTRACT = 512
DEFAULT_NMS = 8
SMOKE_PAIRS = [0, 3]
SMOKE_ANGLES = [0, 90, 180, 270]
FULL_PAIRS = [0, 1, 3, 16]
FULL_ANGLES = list(range(0, 360, 30))


def load_sp_model(weights_path: Path | str, device: torch.device, nms_radius: int = 8):
    from superpoint_pytorch import SuperPoint
    from training import DEFAULT_WEIGHTS

    path = Path(weights_path) if weights_path else Path(DEFAULT_WEIGHTS)
    model = SuperPoint(
        nms_radius=int(nms_radius),
        max_num_keypoints=2048,
        detection_threshold=0.015,
    )
    sd = torch.load(str(path), map_location=device)
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def extract_feats(model, gray: np.ndarray, device: torch.device, resize: int | None):
    img = gray
    if resize is not None:
        h, w = img.shape[:2]
        scale = float(resize) / float(max(h, w))
        if scale != 1.0:
            img = cv2.resize(
                img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
            )
    t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    out = model({"image": t}, training=False)
    kpts = out["keypoints"][0]
    scores = out["keypoint_scores"][0]
    desc = out["descriptors"][0]
    h, w = img.shape[:2]
    if resize is not None and gray.shape[:2] != img.shape[:2]:
        sx = gray.shape[1] / float(w)
        sy = gray.shape[0] / float(h)
        kpts = kpts.clone()
        kpts[:, 0] *= sx
        kpts[:, 1] *= sy
        h, w = gray.shape[:2]
    feats = {
        "keypoints": kpts.unsqueeze(0),
        "keypoint_scores": scores.unsqueeze(0),
        "descriptors": desc.unsqueeze(0),
        "image_size": torch.tensor([[w, h]], device=device, dtype=torch.float32),
    }
    return feats, (w, h)


def match_lg(feats0, feats1, device: torch.device, hp: dict | None = None):
    from lightglue import LightGlue

    hp = hp or {}
    matcher = (
        LightGlue(
            features="superpoint",
            depth_confidence=float(hp.get("lg_depth_confidence", -1.0)),
            width_confidence=float(hp.get("lg_width_confidence", -1.0)),
        )
        .eval()
        .to(device)
    )
    with torch.no_grad():
        m = matcher({"image0": feats0, "image1": feats1})
    from lightglue.utils import rbd

    m = rbd(m)
    matches = m["matches"]
    if matches is None or len(matches) == 0:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=float),
        )
    k0 = feats0["keypoints"][0].detach().cpu().numpy()
    k1 = feats1["keypoints"][0].detach().cpu().numpy()
    mk = matches.detach().cpu().numpy()
    scores = m.get("scores")
    if scores is not None:
        sc = scores.detach().cpu().numpy()
    else:
        sc = np.ones(len(mk), dtype=float)
    return k0[mk[:, 0]], k1[mk[:, 1]], sc


def run_cell(
    model,
    matcher_hp: dict,
    device: torch.device,
    pair_id: int,
    angle: int,
    gt: dict,
    *,
    preview_level: int = 2,
    extract_resize: int = DEFAULT_EXTRACT,
) -> dict:
    import crop_core

    he = crop_core.whole_gray(pair_id, "he", preview_level)
    ihc = crop_core.whole_gray(pair_id, "ihc", preview_level)
    if he is None or ihc is None:
        raise RuntimeError(f"missing preview pair={pair_id}")
    ihc_prerot, pre_M = rigid_sp_lg._rotate_gray(ihc, float(angle))
    f0, (w, h) = extract_feats(model, he, device, extract_resize)
    f1, _ = extract_feats(model, ihc_prerot, device, extract_resize)
    he_pts, ihc_pts, _scores = match_lg(f0, f1, device, matcher_hp)
    if len(he_pts) == 0:
        raise RuntimeError("no matches")
    R_px, t_px, inlier_mask, stats = rigid_sp_lg.fit_rigid_kabsch(
        he_pts, ihc_pts, float(matcher_hp.get("rigid_inlier_px", 3.0))
    )
    rigid_final = rigid_sp_lg._compose_norm_rigid(pre_M, R_px, t_px, w, h)
    cmp_m = bench.compare_rigid_to_gt(rigid_final, gt.get("rigid"), width=float(w), height=float(h))
    tr = cmp_m.get("trans_err_px")
    min_wh = min(float(w), float(h))
    trans_rel = float(tr) / min_wh if tr is not None and min_wh > 0 else None
    rot = cmp_m.get("rot_err_deg")
    ok = (
        rot is not None
        and trans_rel is not None
        and float(rot) <= MAX_ROT_ERR_DEG
        and float(trans_rel) <= MAX_TRANS_ERR_REL
    )
    return {
        "pair_id": int(pair_id),
        "angle": int(angle),
        "n_inliers": int(stats.get("n_inliers") or 0),
        "rot_err_deg": rot,
        "trans_err_rel": trans_rel,
        "auto_pass": bool(ok),
        "error": None,
    }


def evaluate_b1(
    weights_path: Path | str | None,
    *,
    pairs: list[int],
    angles: list[int],
    extract_resize: int = DEFAULT_EXTRACT,
    nms: int = DEFAULT_NMS,
    preview_level: int = 2,
    dataset: str = "muromi",
) -> dict:
    from setup import datasets

    datasets.set_active_dataset(dataset)
    device = lam_sp_lg.device_auto()
    model = load_sp_model(weights_path or conf.resolve("introducing_superpoint/superpoint_v6_from_tf.pth"), device, nms)
    hp = {**rigid_sp_lg.DEFAULT_HYPERPARAMS, "sp_nms_dist": int(nms)}
    cells = []
    n_pass = 0
    n_err = 0
    by_angle: dict[str, dict] = {str(a): {"n": 0, "n_pass": 0} for a in angles}
    for pid in pairs:
        gt = bench.ensure_gt_rigid(pid, dataset)
        for ang in angles:
            try:
                cell = run_cell(
                    model,
                    hp,
                    device,
                    pid,
                    ang,
                    gt,
                    preview_level=preview_level,
                    extract_resize=extract_resize,
                )
            except Exception as e:
                cell = {
                    "pair_id": pid,
                    "angle": ang,
                    "auto_pass": False,
                    "error": str(e),
                }
                n_err += 1
            cells.append(cell)
            by_angle[str(ang)]["n"] += 1
            if cell.get("auto_pass"):
                n_pass += 1
                by_angle[str(ang)]["n_pass"] += 1
    n_total = len(cells)
    for st in by_angle.values():
        st["pass_rate"] = (st["n_pass"] / st["n"]) if st["n"] else None
    return {
        "n_pass": n_pass,
        "n_total": n_total,
        "n_error": n_err,
        "pass_rate": (n_pass / n_total) if n_total else None,
        "by_angle": by_angle,
        "cells": cells,
        "extract_resize": extract_resize,
        "sp_nms_dist": nms,
        "pairs": pairs,
        "angles": angles,
        "gate": {"max_rot_err_deg": MAX_ROT_ERR_DEG, "max_trans_err_rel": MAX_TRANS_ERR_REL},
    }


def smoke_eval(weights_path: Path | str | None, **kw) -> dict:
    return evaluate_b1(
        weights_path,
        pairs=kw.get("pairs") or SMOKE_PAIRS,
        angles=kw.get("angles") or SMOKE_ANGLES,
        extract_resize=int(kw.get("extract_resize") or DEFAULT_EXTRACT),
        nms=int(kw.get("nms") or DEFAULT_NMS),
    )


def full_eval(weights_path: Path | str | None, **kw) -> dict:
    return evaluate_b1(
        weights_path,
        pairs=kw.get("pairs") or FULL_PAIRS,
        angles=kw.get("angles") or FULL_ANGLES,
        extract_resize=int(kw.get("extract_resize") or DEFAULT_EXTRACT),
        nms=int(kw.get("nms") or DEFAULT_NMS),
    )
