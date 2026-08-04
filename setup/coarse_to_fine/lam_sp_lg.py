"""
Shared SuperPoint + LightGlue helpers for rigid prealignment and the C2F LAM.

Extract resolution is explicit: pass resize=None to keep the native array size
(C2F tiles are already CNN 512×344). Pass an int for long-edge resize (rigid
whole-slide previews). Never rely on LightGlue's default 1024.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

DEFAULT_EXTRACT_RESIZE: int | None = None

DEFAULT_HYPERPARAMS = {
    "sp_conf_thresh": 0.015,
    "sp_nms_dist": 4,
    "sp_max_keypoints": 1024,
    "lg_depth_confidence": -1.0,
    "lg_width_confidence": -1.0,
    "match_min_score": 0.1,
    "trans_inlier_px": 3.0,
}


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models(hyperparams: dict | None = None, device: torch.device | None = None):
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    device = device or device_auto()
    from lightglue import LightGlue, SuperPoint

    extractor = (
        SuperPoint(
            max_num_keypoints=int(hp["sp_max_keypoints"]),
            detection_threshold=float(hp["sp_conf_thresh"]),
            nms_radius=int(hp["sp_nms_dist"]),
        )
        .eval()
        .to(device)
    )
    matcher = (
        LightGlue(
            features="superpoint",
            depth_confidence=float(hp["lg_depth_confidence"]),
            width_confidence=float(hp["lg_width_confidence"]),
        )
        .eval()
        .to(device)
    )
    return extractor, matcher, device, hp


def gray_to_torch(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    from lightglue.utils import numpy_image_to_torch

    rgb = np.stack([gray, gray, gray], axis=-1)
    return numpy_image_to_torch(rgb).to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def extract_and_match(
    he: np.ndarray,
    ihc: np.ndarray,
    extractor,
    matcher,
    device: torch.device,
    *,
    resize: int | None = DEFAULT_EXTRACT_RESIZE,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Match two uint8 HxW greyscale arrays.

    resize: None → no online resize (native HxW keypoints).
            int  → LightGlue long-edge resize to that length.
    on_stage: optional callable(str) for coarse progress
              (superpoint_he → superpoint_ihc → lightglue).
    """
    from lightglue.utils import rbd

    def _stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    extract_kw = {"resize": resize}
    empty = {
        "he_pts": np.zeros((0, 2), dtype=float),
        "ihc_pts": np.zeros((0, 2), dtype=float),
        "scores": np.zeros((0,), dtype=float),
        "timing": {
            "sp_he_s": 0.0,
            "sp_ihc_s": 0.0,
            "sp_s": 0.0,
            "lg_s": 0.0,
        },
    }

    with torch.no_grad():
        _stage("superpoint_he")
        _sync(device)
        t0 = time.perf_counter()
        feats0 = extractor.extract(gray_to_torch(he, device), **extract_kw)
        _sync(device)
        sp_he_s = time.perf_counter() - t0

        _stage("superpoint_ihc")
        _sync(device)
        t0 = time.perf_counter()
        feats1 = extractor.extract(gray_to_torch(ihc, device), **extract_kw)
        _sync(device)
        sp_ihc_s = time.perf_counter() - t0

        _stage("lightglue")
        _sync(device)
        t0 = time.perf_counter()
        matches01 = matcher({"image0": feats0, "image1": feats1})
        _sync(device)
        lg_s = time.perf_counter() - t0

    timing = {
        "sp_he_s": float(sp_he_s),
        "sp_ihc_s": float(sp_ihc_s),
        "sp_s": float(sp_he_s + sp_ihc_s),
        "lg_s": float(lg_s),
    }

    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
    matches = matches01["matches"]
    if matches is None or len(matches) == 0:
        empty["timing"] = timing
        return empty

    k0 = feats0["keypoints"].detach().cpu().numpy()
    k1 = feats1["keypoints"].detach().cpu().numpy()
    scores = matches01.get("scores")
    if scores is not None:
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.ones(len(matches), dtype=float)

    m = matches.detach().cpu().numpy()
    he_pts = k0[m[:, 0]]
    ihc_pts = k1[m[:, 1]]
    match_scores = scores[: len(m)] if len(scores) == len(m) else np.ones(len(m), dtype=float)
    return {
        "he_pts": np.asarray(he_pts, dtype=float),
        "ihc_pts": np.asarray(ihc_pts, dtype=float),
        "scores": np.asarray(match_scores, dtype=float),
        "timing": timing,
    }


def translation_from_matches(
    he_pts: np.ndarray,
    ihc_pts: np.ndarray,
    scores: np.ndarray,
    *,
    min_score: float = 0.1,
    inlier_px: float = 3.0,
) -> dict[str, Any]:
    """
    One translation residual from correspondences:
    score-filter → translation consensus → mean displacement of inliers.
    Displacement is HE − IHC in the crop's pixel space (same sign as FFT dx/dy).
    """
    n = min(len(he_pts), len(ihc_pts), len(scores))
    if n == 0:
        return {
            "dx": 0.0,
            "dy": 0.0,
            "psr": 0.0,
            "inliers": np.zeros((0,), dtype=bool),
            "n_matches": 0,
            "n_inliers": 0,
        }

    he_pts = np.asarray(he_pts[:n], dtype=float)
    ihc_pts = np.asarray(ihc_pts[:n], dtype=float)
    scores = np.asarray(scores[:n], dtype=float)
    disps = he_pts - ihc_pts

    scored = scores >= float(min_score)
    if not np.any(scored):
        scored = np.ones(n, dtype=bool)

    work = disps[scored]
    med = np.median(work, axis=0)
    resid = np.linalg.norm(work - med, axis=1)
    local_in = resid <= float(inlier_px)
    if int(local_in.sum()) < 1:
        local_in = np.ones(len(work), dtype=bool)

    mean_disp = work[local_in].mean(axis=0)
    inliers = np.zeros(n, dtype=bool)
    scored_idx = np.flatnonzero(scored)
    inliers[scored_idx[local_in]] = True
    n_in = int(inliers.sum())
    med_score = float(np.median(scores[inliers])) if n_in else 0.0
    psr = float(n_in) * med_score
    return {
        "dx": float(mean_disp[0]),
        "dy": float(mean_disp[1]),
        "psr": psr,
        "inliers": inliers,
        "n_matches": int(n),
        "n_inliers": n_in,
    }


def match_tile_residual(
    he: np.ndarray,
    ihc: np.ndarray,
    extractor,
    matcher,
    device: torch.device,
    hyperparams: dict | None = None,
    *,
    resize: int | None = DEFAULT_EXTRACT_RESIZE,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """C2F LAM: match a tile pair → residual dx/dy + psr (+ match dump)."""
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    raw = extract_and_match(
        he, ihc, extractor, matcher, device, resize=resize, on_stage=on_stage
    )
    summary = translation_from_matches(
        raw["he_pts"],
        raw["ihc_pts"],
        raw["scores"],
        min_score=float(hp["match_min_score"]),
        inlier_px=float(hp["trans_inlier_px"]),
    )
    inliers = summary["inliers"]
    matches = []
    for i in range(summary["n_matches"]):
        matches.append({
            "he": [float(raw["he_pts"][i, 0]), float(raw["he_pts"][i, 1])],
            "ihc": [float(raw["ihc_pts"][i, 0]), float(raw["ihc_pts"][i, 1])],
            "score": float(raw["scores"][i]),
            "inlier": bool(inliers[i]) if i < len(inliers) else False,
        })
    return {
        **summary,
        "he_pts": raw["he_pts"],
        "ihc_pts": raw["ihc_pts"],
        "scores": raw["scores"],
        "matches": matches,
        "timing": raw.get("timing") or {
            "sp_he_s": 0.0,
            "sp_ihc_s": 0.0,
            "sp_s": 0.0,
            "lg_s": 0.0,
        },
    }
