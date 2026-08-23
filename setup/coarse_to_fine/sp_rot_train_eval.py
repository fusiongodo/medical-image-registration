"""Rigid rotational eval using introducing_superpoint weights + LightGlue matcher."""

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
DEFAULT_NN_THRESH = 0.7
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


def build_lg_matcher(device: torch.device, hp: dict | None = None):
    from lightglue import LightGlue

    hp = hp or {}
    return (
        LightGlue(
            features="superpoint",
            depth_confidence=float(hp.get("lg_depth_confidence", -1.0)),
            width_confidence=float(hp.get("lg_width_confidence", -1.0)),
        )
        .eval()
        .to(device)
    )


def match_lg(feats0, feats1, device: torch.device, hp: dict | None = None, matcher=None):
    if matcher is None:
        matcher = build_lg_matcher(device, hp)
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


def match_nn_two_way(feats0, feats1, nn_thresh: float = DEFAULT_NN_THRESH):
    """
    feats0/feats1: keypoints [1,N,2], unit descriptors [1,N,D] or [1,D,N].
    Returns (pts0 [M,2], pts1 [M,2], l2_dist [M]) after mutual NN + threshold.
    """
    empty = (
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 2), dtype=float),
        np.zeros((0,), dtype=float),
    )
    k0 = feats0["keypoints"][0].detach().cpu().numpy()
    k1 = feats1["keypoints"][0].detach().cpu().numpy()
    d0 = feats0["descriptors"][0].detach().cpu().float().numpy()
    d1 = feats1["descriptors"][0].detach().cpu().float().numpy()
    if d0.size == 0 or d1.size == 0 or d0.ndim != 2 or d1.ndim != 2:
        return empty
    if d0.shape[-1] == d1.shape[-1]:
        desc1 = d0.T
        desc2 = d1.T
    else:
        desc1 = d0
        desc2 = d1
    if desc1.shape[0] != desc2.shape[0] or desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return empty
    if nn_thresh < 0.0:
        raise ValueError("nn_thresh should be non-negative")
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2.0 - 2.0 * np.clip(dmat, -1.0, 1.0))
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    keep = scores < float(nn_thresh)
    idx2 = np.argmin(dmat, axis=0)
    keep = np.logical_and(keep, np.arange(len(idx)) == idx2[idx])
    if not np.any(keep):
        return empty
    i0 = np.arange(desc1.shape[1])[keep]
    i1 = idx[keep]
    return k0[i0], k1[i1], scores[keep]


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


def evaluate_slides(
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
    return evaluate_slides(
        weights_path,
        pairs=kw.get("pairs") or SMOKE_PAIRS,
        angles=kw.get("angles") or SMOKE_ANGLES,
        extract_resize=int(kw.get("extract_resize") or DEFAULT_EXTRACT),
        nms=int(kw.get("nms") or DEFAULT_NMS),
    )


def full_eval(weights_path: Path | str | None, **kw) -> dict:
    return evaluate_slides(
        weights_path,
        pairs=kw.get("pairs") or FULL_PAIRS,
        angles=kw.get("angles") or FULL_ANGLES,
        extract_resize=int(kw.get("extract_resize") or DEFAULT_EXTRACT),
        nms=int(kw.get("nms") or DEFAULT_NMS),
    )


def _angle_from_R(R: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0]))))


def _angle_err_deg(a: float, b: float) -> float:
    return abs(((a - b) + 180.0) % 360.0 - 180.0)


def _centre_residual_rel(R_px, t_px, w: float, h: float):
    """
    How far the fitted rigid transform moves the image centre, relative to the short
    side. The synthetic warp rotates about the centre, so the centre is a fixed point
    and a perfect fit scores 0. Do not use |t_px| here: for a centre rotation
    t = (I - R)c, which reaches 1.41 of the short side at 180 deg however good the fit.
    """
    min_wh = min(float(w), float(h))
    if min_wh <= 0:
        return None
    c = np.array([float(w) / 2.0, float(h) / 2.0], dtype=np.float64)
    R = np.asarray(R_px, dtype=np.float64).reshape(2, 2)
    t = np.asarray(t_px, dtype=np.float64).reshape(2)
    return float(np.linalg.norm(R @ c + t - c)) / min_wh


def run_tile_cell(
    model,
    matcher_hp: dict,
    device: torch.device,
    page: np.ndarray,
    tile: dict,
    angle: float,
    *,
    depth: int,
    preview_level: int,
    src_size: int,
    out_size: int,
    extract_resize: int,
    matcher=None,
    match_fn=None,
) -> dict:
    from setup.coarse_to_fine import sp_rot_train_data as tdata

    base, warped, _valid, _H = tdata.make_warp_pair(
        page,
        tile,
        depth=depth,
        preview_level=preview_level,
        src_size=src_size,
        out_size=out_size,
        theta_deg=float(angle),
    )
    f0, (w, h) = extract_feats(model, base, device, extract_resize)
    f1, _ = extract_feats(model, warped, device, extract_resize)
    if match_fn is not None:
        pts0, pts1, _scores = match_fn(f0, f1)
    else:
        pts0, pts1, _scores = match_lg(f0, f1, device, matcher_hp, matcher=matcher)
    n_matches = int(len(pts0))
    if n_matches < 2:
        raise RuntimeError("need at least 2 matches")
    R_px, t_px, _inl, stats = rigid_sp_lg.fit_rigid_kabsch(
        pts0, pts1, float(matcher_hp.get("rigid_inlier_px", 3.0))
    )
    pred_ang = _angle_from_R(R_px)
    rot_err = _angle_err_deg(pred_ang, float(angle))
    trans_rel = _centre_residual_rel(R_px, t_px, w, h)
    ok = (
        trans_rel is not None
        and rot_err <= MAX_ROT_ERR_DEG
        and float(trans_rel) <= MAX_TRANS_ERR_REL
    )
    return {
        "pair_id": int(tile["pair_id"]),
        "loc": tile.get("loc"),
        "angle": float(angle),
        "n_matches": n_matches,
        "n_inliers": int(stats.get("n_inliers") or 0),
        "rot_err_deg": rot_err,
        "trans_err_rel": trans_rel,
        "auto_pass": bool(ok),
        "error": None,
    }


def _kabsch_gate(pts0, pts1, angle: float, w: float, h: float, inlier_px: float) -> dict:
    n_matches = int(len(pts0))
    if n_matches < 2:
        return {
            "n_matches": n_matches,
            "n_inliers": 0,
            "rot_err_deg": None,
            "trans_err_rel": None,
            "auto_pass": False,
            "error": "need at least 2 matches",
        }
    R_px, t_px, _inl, stats = rigid_sp_lg.fit_rigid_kabsch(pts0, pts1, float(inlier_px))
    rot_err = _angle_err_deg(_angle_from_R(R_px), float(angle))
    trans_rel = _centre_residual_rel(R_px, t_px, w, h)
    ok = (
        trans_rel is not None
        and rot_err <= MAX_ROT_ERR_DEG
        and float(trans_rel) <= MAX_TRANS_ERR_REL
    )
    return {
        "n_matches": n_matches,
        "n_inliers": int(stats.get("n_inliers") or 0),
        "rot_err_deg": rot_err,
        "trans_err_rel": trans_rel,
        "auto_pass": bool(ok),
        "error": None,
    }


def run_tile_cell_matchers(
    model,
    device: torch.device,
    page: np.ndarray,
    tile: dict,
    angle: float,
    *,
    depth: int,
    preview_level: int,
    src_size: int,
    out_size: int,
    extract_resize: int,
    lg_matcher,
    nn_thresh: float = DEFAULT_NN_THRESH,
    inlier_px: float = 3.0,
) -> dict:
    """
    One extract per image, then match twice. Returns {"nn": cell, "lg": cell}
    where each cell has n_matches / n_inliers / rot_err_deg / trans_err_rel /
    auto_pass for the same warp.
    """
    from setup.coarse_to_fine import sp_rot_train_data as tdata

    base, warped, _valid, _H = tdata.make_warp_pair(
        page,
        tile,
        depth=depth,
        preview_level=preview_level,
        src_size=src_size,
        out_size=out_size,
        theta_deg=float(angle),
    )
    f0, (w, h) = extract_feats(model, base, device, extract_resize)
    f1, _ = extract_feats(model, warped, device, extract_resize)
    n_kp = (int(f0["keypoints"].shape[1]), int(f1["keypoints"].shape[1]))
    out = {}
    for kind in ("nn", "lg"):
        try:
            if kind == "nn":
                pts0, pts1, _sc = match_nn_two_way(f0, f1, nn_thresh)
            else:
                pts0, pts1, _sc = match_lg(f0, f1, device, None, matcher=lg_matcher)
            cell = _kabsch_gate(pts0, pts1, angle, w, h, inlier_px)
        except Exception as e:
            cell = {
                "n_matches": None,
                "n_inliers": None,
                "rot_err_deg": None,
                "trans_err_rel": None,
                "auto_pass": False,
                "error": str(e),
            }
        cell.update(
            {
                "pair_id": int(tile["pair_id"]),
                "loc": tile.get("loc"),
                "angle": float(angle),
                "n_kp0": n_kp[0],
                "n_kp1": n_kp[1],
            }
        )
        out[kind] = cell
    return out


def evaluate_tile_matchers(
    model,
    tiles: list[dict],
    *,
    angles: list[float],
    device: torch.device,
    extract_resize: int = DEFAULT_EXTRACT,
    nms: int = DEFAULT_NMS,
    depth: int = 5,
    preview_level: int = 2,
    src_size: int = 768,
    out_size: int = 512,
    dataset: str = "muromi",
    nn_thresh: float = DEFAULT_NN_THRESH,
) -> dict:
    """
    Same warps evaluated with two-way NN and LightGlue on one shared extract.
    Returns {"nn": {...aggregate + cells}, "lg": {...}} with per-matcher k/n.
    """
    from setup import datasets

    datasets.set_active_dataset(dataset)
    import crop_core

    model = model.to(device).eval()
    hp = {**rigid_sp_lg.DEFAULT_HYPERPARAMS, "sp_nms_dist": int(nms)}
    lg_matcher = build_lg_matcher(device, hp)
    inlier_px = float(hp.get("rigid_inlier_px", 3.0))
    agg = {
        kind: {"cells": [], "by_angle": {str(a): {"n": 0, "n_pass": 0} for a in angles}}
        for kind in ("nn", "lg")
    }
    page_cache: dict[int, np.ndarray] = {}
    for tile in tiles:
        pid = int(tile["pair_id"])
        if pid not in page_cache:
            g = crop_core.whole_gray(pid, "he", preview_level)
            if g is None:
                raise RuntimeError(f"missing page pair={pid}")
            page_cache[pid] = g
        for ang in angles:
            cells = run_tile_cell_matchers(
                model,
                device,
                page_cache[pid],
                tile,
                float(ang),
                depth=depth,
                preview_level=preview_level,
                src_size=src_size,
                out_size=out_size,
                extract_resize=extract_resize,
                lg_matcher=lg_matcher,
                nn_thresh=nn_thresh,
                inlier_px=inlier_px,
            )
            for kind, cell in cells.items():
                agg[kind]["cells"].append(cell)
                st = agg[kind]["by_angle"][str(ang)]
                st["n"] += 1
                if cell.get("auto_pass"):
                    st["n_pass"] += 1
    out = {}
    for kind, a in agg.items():
        cells = a["cells"]
        n_pass = sum(1 for c in cells if c.get("auto_pass"))
        n_err = sum(1 for c in cells if c.get("error"))
        for st in a["by_angle"].values():
            st["pass_rate"] = (st["n_pass"] / st["n"]) if st["n"] else None
        out[kind] = {
            "n_pass": n_pass,
            "n_total": len(cells),
            "pass_rate": (n_pass / len(cells)) if cells else None,
            "n_error": n_err,
            "by_angle": a["by_angle"],
            "cells": cells,
        }
    out["angles"] = list(angles)
    out["extract_resize"] = extract_resize
    out["sp_nms_dist"] = nms
    out["nn_thresh"] = float(nn_thresh)
    out["gate"] = {
        "max_rot_err_deg": MAX_ROT_ERR_DEG,
        "max_trans_err_rel": MAX_TRANS_ERR_REL,
    }
    return out


def evaluate_tiles(
    weights_path: Path | str | None,
    tiles: list[dict],
    *,
    angles: list[float] | None = None,
    extract_resize: int = DEFAULT_EXTRACT,
    nms: int = DEFAULT_NMS,
    depth: int = 5,
    preview_level: int = 2,
    src_size: int = 768,
    out_size: int = 512,
    max_tiles: int | None = 24,
    split_seed: int = 0,
    dataset: str = "muromi",
    model=None,
    device: torch.device | None = None,
    on_progress=None,
    match_kind: str = "lg",
    nn_thresh: float = DEFAULT_NN_THRESH,
) -> dict:
    from setup import datasets
    from setup.coarse_to_fine import sp_rot_train_data as tdata

    datasets.set_active_dataset(dataset)
    angles = list(angles if angles is not None else FULL_ANGLES)
    use_tiles = tdata.subsample_tiles(tiles, max_tiles, split_seed)
    if device is None:
        device = lam_sp_lg.device_auto()
    if model is None:
        model = load_sp_model(
            weights_path or conf.resolve("introducing_superpoint/superpoint_v6_from_tf.pth"),
            device,
            nms,
        )
    else:
        model = model.to(device).eval()
    hp = {**rigid_sp_lg.DEFAULT_HYPERPARAMS, "sp_nms_dist": int(nms)}
    kind = str(match_kind or "lg")
    if kind == "lg":
        matcher = build_lg_matcher(device, hp)
        match_fn = None
    elif kind == "nn":
        matcher = None
        thresh = float(nn_thresh)

        def match_fn(f0, f1, _t=thresh):
            return match_nn_two_way(f0, f1, _t)
    else:
        raise ValueError(f"unknown match_kind {kind!r}")
    page_cache: dict[tuple[int, str], np.ndarray] = {}
    cells = []
    n_pass = 0
    n_err = 0
    by_angle: dict[str, dict] = {str(a): {"n": 0, "n_pass": 0} for a in angles}
    n_total_plan = max(1, len(use_tiles) * len(angles))

    import crop_core

    for ti, tile in enumerate(use_tiles):
        pid = int(tile["pair_id"])
        side = "he"
        key = (pid, side)
        if key not in page_cache:
            g = crop_core.whole_gray(pid, side, preview_level)
            if g is None:
                for ang in angles:
                    cells.append(
                        {
                            "pair_id": pid,
                            "loc": tile.get("loc"),
                            "angle": float(ang),
                            "auto_pass": False,
                            "error": "missing page",
                        }
                    )
                    n_err += 1
                    by_angle[str(ang)]["n"] += 1
                continue
            page_cache[key] = g
        page = page_cache[key]
        for ang in angles:
            try:
                cell = run_tile_cell(
                    model,
                    hp,
                    device,
                    page,
                    tile,
                    float(ang),
                    depth=depth,
                    preview_level=preview_level,
                    src_size=src_size,
                    out_size=out_size,
                    extract_resize=extract_resize,
                    matcher=matcher,
                    match_fn=match_fn,
                )
            except Exception as e:
                cell = {
                    "pair_id": pid,
                    "loc": tile.get("loc"),
                    "angle": float(ang),
                    "auto_pass": False,
                    "error": str(e),
                }
                n_err += 1
            cells.append(cell)
            by_angle[str(ang)]["n"] += 1
            if cell.get("auto_pass"):
                n_pass += 1
                by_angle[str(ang)]["n_pass"] += 1
            if on_progress is not None and len(cells) % 12 == 0:
                on_progress(len(cells), n_total_plan, n_pass)
    n_total = len(cells)
    for st in by_angle.values():
        st["pass_rate"] = (st["n_pass"] / st["n"]) if st["n"] else None
    return {
        "n_pass": n_pass,
        "n_total": n_total,
        "n_error": n_err,
        "pass_rate": (n_pass / n_total) if n_total else None,
        "by_angle": by_angle,
        "n_tiles": len(use_tiles),
        "angles": angles,
        "extract_resize": extract_resize,
        "sp_nms_dist": nms,
        "match_kind": kind,
        "nn_thresh": float(nn_thresh) if kind == "nn" else None,
        "gate": {"max_rot_err_deg": MAX_ROT_ERR_DEG, "max_trans_err_rel": MAX_TRANS_ERR_REL},
        "cells": cells,
    }
