"""
SuperPoint + LightGlue rigid pre-alignment (light_v1).

Approved transform: data/rigid/light_v1/{pair}.json
Run scratch:        data/rigid/light_v1/{pair}/run/

`rigid` is a 2×3 matrix in normalised [0,1]² mapping IHC → HE:
    [hx, hy]^T = [[r00, r01, tx], [r10, r11, ty]] @ [ix, iy, 1]^T

Pre-rotation on the moving IHC is composed into the saved matrix so crop-time
only needs `rigid`. Deskew (if any) is applied after rigid at crop time.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))
import conf

from setup.coarse_to_fine import lam_sp_lg
from setup.coarse_to_fine.identity import pair_fingerprint
from setup.coarse_to_fine.reg_branches import clear_lam_caches

VERSION = "light_v1"
RIGID_ROOT = conf.PROJECT_ROOT / "data" / "rigid" / VERSION

DEFAULT_HYPERPARAMS = {
    "sp_conf_thresh": 0.015,
    "sp_nms_dist": 4,
    "sp_max_keypoints": 2048,
    "lg_depth_confidence": -1.0,
    "lg_width_confidence": -1.0,
    "rigid_inlier_px": 3.0,
}
DEFAULT_PREVIEW_LEVEL = 2
RIGID_EXTRACT_RESIZE = 1024


def saved_path(pair_id: int) -> Path:
    return RIGID_ROOT / f"{pair_id}.json"


def matches_path(pair_id: int) -> Path:
    return RIGID_ROOT / f"{pair_id}.matches.json"


def run_dir(pair_id: int) -> Path:
    return RIGID_ROOT / str(pair_id) / "run"


def load(pair_id: int) -> dict | None:
    path = saved_path(pair_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_matches(pair_id: int) -> dict | None:
    path = matches_path(pair_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_matches(pair_id: int, store: dict | None) -> None:
    path = matches_path(pair_id)
    if not store:
        if path.exists():
            path.unlink()
        return
    RIGID_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, separators=(",", ":")))


def clear(pair_id: int) -> None:
    path = saved_path(pair_id)
    if path.exists():
        path.unlink()
    write_matches(pair_id, None)


def write(pair_id: int, store: dict | None) -> None:
    if not store:
        clear(pair_id)
        return
    RIGID_ROOT.mkdir(parents=True, exist_ok=True)
    saved_path(pair_id).write_text(json.dumps(store, separators=(",", ":")))


def _normalise_run_matches(result: dict, matches_raw: dict) -> dict:
    w = float(result.get("stats", {}).get("width") or 1.0)
    h = float(result.get("stats", {}).get("height") or 1.0)
    he = matches_raw.get("he") or []
    ihc = matches_raw.get("ihc") or []
    scores = matches_raw.get("scores") or []
    inliers = matches_raw.get("inliers") or []
    points = []
    n = min(len(he), len(ihc))
    for i in range(n):
        hx, hy = float(he[i][0]), float(he[i][1])
        ix, iy = float(ihc[i][0]), float(ihc[i][1])
        points.append({
            "he": [hx / w, hy / h],
            "ihc": [ix / w, iy / h],
            "score": float(scores[i]) if i < len(scores) else 1.0,
            "inlier": bool(inliers[i]) if i < len(inliers) else True,
        })
    return {
        "pair_id": int(result["pair_id"]),
        "version": VERSION,
        "identity": result.get("identity") or pair_fingerprint(int(result["pair_id"])),
        "preview_level": int(result["preview_level"]),
        "width": w,
        "height": h,
        "pre_rotation_deg": float(result.get("pre_rotation_deg") or 0.0),
        "rigid": result.get("rigid"),
        "rigid_prerot": result.get("rigid_prerot"),
        "points": points,
        "n_matches": len(points),
        "n_inliers": sum(1 for p in points if p["inlier"]),
        "saved_at": int(time.time()),
    }


def clear_caches(pair_id: int) -> int:
    return clear_lam_caches(pair_id)


def _progress(pair_id: int, stage: str, detail: str = "") -> None:
    rd = run_dir(pair_id)
    rd.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "detail": detail, "ts": time.time()}
    (rd / "progress.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"stage={stage} {detail}".rstrip(), flush=True)


def _rotate_gray(img: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate about centre; return (warped, 2×3 pixel-space matrix for original→rotated)."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    warped = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    return warped, M


def _norm_from_pixel_rigid(R_px: np.ndarray, t_px: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert pixel-space he = R @ ihc + t into normalised [0,1]² 2×3."""
    S = np.diag([float(w), float(h)])
    Sinv = np.diag([1.0 / float(w), 1.0 / float(h)])
    R_n = Sinv @ R_px @ S
    t_n = Sinv @ t_px
    return np.array(
        [[R_n[0, 0], R_n[0, 1], t_n[0]], [R_n[1, 0], R_n[1, 1], t_n[1]]],
        dtype=float,
    )


def _compose_norm_rigid(
    pre_rot_px: np.ndarray,
    R_prerot_px: np.ndarray,
    t_prerot_px: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Compose pixel pre-rotation (original IHC → prerot) with pixel rigid
    (prerot → HE) into a normalised rigid (original IHC → HE).
    Composition stays in pixel space so centre-rotation is exact on non-square images.
    """
    A_px = pre_rot_px[:, :2].astype(float)
    b_px = pre_rot_px[:, 2].astype(float)
    R_out = R_prerot_px @ A_px
    t_out = R_prerot_px @ b_px + t_prerot_px
    return _norm_from_pixel_rigid(R_out, t_out, w, h)


def fit_rigid_kabsch(
    he_xy: np.ndarray,
    ihc_xy: np.ndarray,
    inlier_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Fit rigid he ≈ R @ ihc + t in pixel coords; refine with inlier filter.
    Returns (R_px 2×2, t_px 2, inlier_mask, stats).
    """
    if len(he_xy) < 2:
        raise ValueError("need at least 2 matches for rigid fit")

    def _fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c_src = src.mean(axis=0)
        c_dst = dst.mean(axis=0)
        X = src - c_src
        Y = dst - c_dst
        U, _, Vt = np.linalg.svd(X.T @ Y)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = c_dst - R @ c_src
        return R, t

    R, t = _fit(ihc_xy, he_xy)
    pred = (R @ ihc_xy.T).T + t
    resid = np.linalg.norm(pred - he_xy, axis=1)
    mask = resid <= inlier_px
    if int(mask.sum()) >= 2:
        R, t = _fit(ihc_xy[mask], he_xy[mask])
        pred = (R @ ihc_xy.T).T + t
        resid = np.linalg.norm(pred - he_xy, axis=1)
        mask = resid <= inlier_px

    n_in = int(mask.sum())
    rmse = float(np.sqrt((resid[mask] ** 2).mean())) if n_in else float("nan")
    rot_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    stats = {
        "rotation_deg": rot_deg,
        "tx_px": float(t[0]),
        "ty_px": float(t[1]),
        "rmse_px": rmse,
        "n_inliers": n_in,
    }
    return R, t, mask, stats


def _draw_matches(
    he: np.ndarray,
    ihc: np.ndarray,
    he_pts: np.ndarray,
    ihc_pts: np.ndarray,
    inliers: np.ndarray | None = None,
) -> np.ndarray:
    h = max(he.shape[0], ihc.shape[0])
    canvas = np.full((h, he.shape[1] + ihc.shape[1]), 255, dtype=np.uint8)
    canvas[: he.shape[0], : he.shape[1]] = he
    canvas[: ihc.shape[0], he.shape[1] : he.shape[1] + ihc.shape[1]] = ihc
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    n = len(he_pts)
    for i in range(n):
        p0 = (int(round(he_pts[i, 0])), int(round(he_pts[i, 1])))
        p1 = (int(round(ihc_pts[i, 0] + he.shape[1])), int(round(ihc_pts[i, 1])))
        ok = True if inliers is None else bool(inliers[i])
        color = (40, 200, 80) if ok else (80, 80, 200)
        cv2.circle(rgb, p0, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(rgb, p1, 2, color, -1, lineType=cv2.LINE_AA)
        if ok or inliers is None:
            cv2.line(rgb, p0, p1, color, 1, lineType=cv2.LINE_AA)
    return rgb


def _apply_rigid_preview(ihc: np.ndarray, rigid_n: np.ndarray) -> np.ndarray:
    h, w = ihc.shape[:2]
    r00, r01, tx = rigid_n[0]
    r10, r11, ty = rigid_n[1]
    S = np.diag([float(w), float(h)])
    Sinv = np.diag([1.0 / float(w), 1.0 / float(h)])
    R_n = np.array([[r00, r01], [r10, r11]], dtype=float)
    t_n = np.array([tx, ty], dtype=float)
    R_px = S @ R_n @ Sinv
    t_px = S @ t_n
    Rinv = np.linalg.inv(R_px)
    b = -Rinv @ t_px
    M = np.array([[Rinv[0, 0], Rinv[0, 1], b[0]], [Rinv[1, 0], Rinv[1, 1], b[1]]], dtype=np.float64)
    return cv2.warpAffine(
        ihc, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )


def run(
    pair_id: int,
    preview_level: int = DEFAULT_PREVIEW_LEVEL,
    pre_rotation_deg: float = 0.0,
    hyperparams: dict | None = None,
    extract_resize: int | None = None,
    write_artifacts: bool = True,
) -> dict:
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    resize = RIGID_EXTRACT_RESIZE if extract_resize is None else int(extract_resize)
    rd = run_dir(pair_id)
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True, exist_ok=True)

    _progress(pair_id, "load", "loading whole greyscale")
    from crop_core import whole_gray

    he = whole_gray(pair_id, "he", preview_level)
    ihc = whole_gray(pair_id, "ihc", preview_level)
    if he is None or ihc is None:
        raise RuntimeError(f"no whole preview for pair {pair_id} level {preview_level}")

    if write_artifacts:
        cv2.imwrite(str(rd / "he.png"), he)
        cv2.imwrite(str(rd / "ihc.png"), ihc)

    _progress(pair_id, "prerot", f"pre_rotation_deg={pre_rotation_deg}")
    ihc_prerot, pre_M = _rotate_gray(ihc, float(pre_rotation_deg))
    if write_artifacts:
        cv2.imwrite(str(rd / "ihc_prerot.png"), ihc_prerot)

    _progress(pair_id, "superpoint", "loading models")
    extractor, matcher, device, _ = lam_sp_lg.build_models(hp)
    _progress(pair_id, "superpoint", f"device={device}")
    _progress(pair_id, "lightglue", "matching")
    raw = lam_sp_lg.extract_and_match(
        he, ihc_prerot, extractor, matcher, device, resize=resize
    )
    he_pts = raw["he_pts"]
    ihc_pts = raw["ihc_pts"]
    match_scores = raw["scores"]
    if len(he_pts) == 0:
        raise RuntimeError("LightGlue returned no matches")

    _progress(pair_id, "fit", f"n_matches={len(he_pts)}")
    h, w = he.shape[:2]
    R_px, t_px, inlier_mask, stats = fit_rigid_kabsch(
        he_pts, ihc_pts, float(hp["rigid_inlier_px"])
    )
    rigid_prerot_n = _norm_from_pixel_rigid(R_px, t_px, w, h)
    rigid_final = _compose_norm_rigid(pre_M, R_px, t_px, w, h)
    stats = {
        **stats,
        "tx": float(rigid_final[0, 2]),
        "ty": float(rigid_final[1, 2]),
        "width": float(w),
        "height": float(h),
        "final_rotation_deg": float(
            np.degrees(np.arctan2(float(R_px[1, 0]), float(R_px[0, 0])))
        ),
    }
    A_px = pre_M[:, :2].astype(float)
    R_final_px = R_px @ A_px
    stats["final_rotation_deg"] = float(
        np.degrees(np.arctan2(float(R_final_px[1, 0]), float(R_final_px[0, 0])))
    )

    if write_artifacts:
        matches_payload = {
            "he": he_pts.tolist(),
            "ihc": ihc_pts.tolist(),
            "scores": [float(s) for s in match_scores],
            "inliers": [bool(x) for x in inlier_mask],
        }
        (rd / "matches.json").write_text(json.dumps(matches_payload, separators=(",", ":")))
        match_viz = _draw_matches(he, ihc_prerot, he_pts, ihc_pts, inlier_mask)
        cv2.imwrite(str(rd / "matches.png"), match_viz)
        _progress(pair_id, "preview", "warping IHC with rigid")
        ihc_rigid = _apply_rigid_preview(ihc, rigid_final)
        cv2.imwrite(str(rd / "ihc_rigid.png"), ihc_rigid)

    result = {
        "pair_id": pair_id,
        "version": VERSION,
        "identity": pair_fingerprint(pair_id),
        "method": "superpoint_lightglue",
        "preview_level": int(preview_level),
        "pre_rotation_deg": float(pre_rotation_deg),
        "extract_resize": int(resize),
        "hyperparams": hp,
        "n_matches": int(len(he_pts)),
        "n_inliers": int(stats["n_inliers"]),
        "rigid": rigid_final.tolist(),
        "rigid_prerot": rigid_prerot_n.tolist(),
        "stats": stats,
        "ran_at": int(time.time()),
    }
    if write_artifacts:
        (rd / "result.json").write_text(json.dumps(result, separators=(",", ":")))
    _progress(pair_id, "done", f"inliers={stats['n_inliers']}")
    return result


def save_from_run(pair_id: int) -> dict:
    rd = run_dir(pair_id)
    result_path = rd / "result.json"
    if not result_path.exists():
        return {"error": "no successful run to save"}
    result = json.loads(result_path.read_text())
    matches_raw_path = rd / "matches.json"
    if not matches_raw_path.exists():
        return {"error": "no run matches to save"}
    matches_raw = json.loads(matches_raw_path.read_text())
    matches_store = _normalise_run_matches(result, matches_raw)
    store = {
        "pair_id": pair_id,
        "version": VERSION,
        "identity": result.get("identity") or pair_fingerprint(pair_id),
        "method": "superpoint_lightglue",
        "preview_level": result["preview_level"],
        "pre_rotation_deg": result["pre_rotation_deg"],
        "hyperparams": result["hyperparams"],
        "n_matches": result["n_matches"],
        "n_inliers": result["n_inliers"],
        "rigid": result["rigid"],
        "rigid_prerot": result.get("rigid_prerot"),
        "stats": result["stats"],
        "saved_at": int(time.time()),
        "matches_file": matches_path(pair_id).name,
    }
    write(pair_id, store)
    write_matches(pair_id, matches_store)
    cleared = clear_caches(pair_id)
    return {
        "ok": True,
        "saved": store,
        "matches": matches_store,
        "caches_cleared": cleared,
    }


def load_run(pair_id: int) -> dict | None:
    path = run_dir(pair_id) / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def reclassify_inliers(pair_id: int, inlier_px: float) -> dict:
    """
    Cheap post-hoc Kabsch + inlier reflag on an existing run (no SuperPoint/LightGlue).
    Rewrites matches, matches.png, ihc_rigid, result; refreshes saved matches if present.
    """
    rd = run_dir(pair_id)
    result_path = rd / "result.json"
    matches_path_run = rd / "matches.json"
    if not result_path.exists() or not matches_path_run.exists():
        raise RuntimeError("no finished run to reclassify")

    result = json.loads(result_path.read_text())
    matches_raw = json.loads(matches_path_run.read_text())
    he_pts = np.asarray(matches_raw["he"], dtype=float)
    ihc_pts = np.asarray(matches_raw["ihc"], dtype=float)
    if len(he_pts) < 2:
        raise RuntimeError("need at least 2 matches to reclassify")

    w = float(result.get("stats", {}).get("width") or 1.0)
    h = float(result.get("stats", {}).get("height") or 1.0)
    pre_rotation_deg = float(result.get("pre_rotation_deg") or 0.0)
    hp = dict(result.get("hyperparams") or DEFAULT_HYPERPARAMS)
    hp["rigid_inlier_px"] = float(inlier_px)

    R_px, t_px, inlier_mask, stats = fit_rigid_kabsch(he_pts, ihc_pts, float(inlier_px))
    rigid_prerot_n = _norm_from_pixel_rigid(R_px, t_px, int(w), int(h))

    he = cv2.imread(str(rd / "he.png"), cv2.IMREAD_GRAYSCALE)
    ihc = cv2.imread(str(rd / "ihc.png"), cv2.IMREAD_GRAYSCALE)
    if he is None or ihc is None:
        from crop_core import whole_gray

        level = int(result.get("preview_level") or DEFAULT_PREVIEW_LEVEL)
        he = whole_gray(pair_id, "he", level)
        ihc = whole_gray(pair_id, "ihc", level)
        if he is None or ihc is None:
            raise RuntimeError("missing preview images for reclassify")
        cv2.imwrite(str(rd / "he.png"), he)
        cv2.imwrite(str(rd / "ihc.png"), ihc)

    ihc_prerot, pre_M = _rotate_gray(ihc, pre_rotation_deg)
    cv2.imwrite(str(rd / "ihc_prerot.png"), ihc_prerot)
    rigid_final = _compose_norm_rigid(pre_M, R_px, t_px, he.shape[1], he.shape[0])
    A_px = pre_M[:, :2].astype(float)
    R_final_px = R_px @ A_px
    stats = {
        **stats,
        "tx": float(rigid_final[0, 2]),
        "ty": float(rigid_final[1, 2]),
        "width": float(he.shape[1]),
        "height": float(he.shape[0]),
        "final_rotation_deg": float(
            np.degrees(np.arctan2(float(R_final_px[1, 0]), float(R_final_px[0, 0])))
        ),
    }

    matches_raw["inliers"] = [bool(x) for x in inlier_mask]
    matches_path_run.write_text(json.dumps(matches_raw, separators=(",", ":")))
    match_viz = _draw_matches(he, ihc_prerot, he_pts, ihc_pts, inlier_mask)
    cv2.imwrite(str(rd / "matches.png"), match_viz)

    ihc_rigid = _apply_rigid_preview(ihc, rigid_final)
    cv2.imwrite(str(rd / "ihc_rigid.png"), ihc_rigid)

    result["hyperparams"] = hp
    result["n_matches"] = int(len(he_pts))
    result["n_inliers"] = int(stats["n_inliers"])
    result["rigid"] = rigid_final.tolist()
    result["rigid_prerot"] = rigid_prerot_n.tolist()
    result["stats"] = stats
    result["reclassified_at"] = int(time.time())
    result_path.write_text(json.dumps(result, separators=(",", ":")))

    saved_refresh = None
    if matches_path(pair_id).exists() or saved_path(pair_id).exists():
        matches_store = _normalise_run_matches(result, matches_raw)
        write_matches(pair_id, matches_store)
        saved = load(pair_id)
        if saved is not None:
            saved["n_matches"] = result["n_matches"]
            saved["n_inliers"] = result["n_inliers"]
            saved["rigid"] = result["rigid"]
            saved["rigid_prerot"] = result["rigid_prerot"]
            saved["stats"] = result["stats"]
            saved["hyperparams"] = hp
            write(pair_id, saved)
        saved_refresh = matches_store

    return {
        "ok": True,
        "inlier_px": float(inlier_px),
        "n_matches": result["n_matches"],
        "n_inliers": result["n_inliers"],
        "run": result,
        "matches": saved_refresh,
    }


def load_progress(pair_id: int) -> dict | None:
    path = run_dir(pair_id) / "progress.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _matches_for_fit(pair_id: int) -> dict:
    saved_m = load_matches(pair_id)
    if saved_m is not None and saved_m.get("points"):
        return saved_m
    result = load_run(pair_id)
    rd = run_dir(pair_id)
    raw_path = rd / "matches.json"
    if result is None or not raw_path.exists():
        raise RuntimeError("no saved or run matches available")
    return _normalise_run_matches(result, json.loads(raw_path.read_text()))


def _ensure_rigid_preview(pair_id: int, matches: dict) -> tuple[np.ndarray, np.ndarray, Path]:
    """Return (he, ihc_rigid, run_dir), regenerating previews if missing."""
    rd = run_dir(pair_id)
    rd.mkdir(parents=True, exist_ok=True)
    he_path = rd / "he.png"
    rigid_path = rd / "ihc_rigid.png"
    level = int(matches.get("preview_level") or DEFAULT_PREVIEW_LEVEL)
    rigid = matches.get("rigid")
    if rigid is None:
        saved = load(pair_id)
        run = load_run(pair_id)
        rigid = (saved or {}).get("rigid") or (run or {}).get("rigid")
    if rigid is None:
        raise RuntimeError("no rigid matrix available for field preview")

    if he_path.exists() and rigid_path.exists():
        he = cv2.imread(str(he_path), cv2.IMREAD_GRAYSCALE)
        ihc_rigid = cv2.imread(str(rigid_path), cv2.IMREAD_GRAYSCALE)
        if he is not None and ihc_rigid is not None:
            return he, ihc_rigid, rd

    from crop_core import whole_gray

    he = whole_gray(pair_id, "he", level)
    ihc = whole_gray(pair_id, "ihc", level)
    if he is None or ihc is None:
        raise RuntimeError(f"no whole preview for pair {pair_id} level {level}")
    ihc_rigid = _apply_rigid_preview(ihc, np.asarray(rigid, dtype=float))
    cv2.imwrite(str(he_path), he)
    cv2.imwrite(str(rigid_path), ihc_rigid)
    return he, ihc_rigid, rd


def _warp_field_preview(moving: np.ndarray, field) -> np.ndarray:
    h_px, w_px = moving.shape[:2]
    ys = np.arange(h_px, dtype=np.float32)
    xs = np.arange(w_px, dtype=np.float32)
    map_x = np.empty((h_px, w_px), dtype=np.float32)
    map_y = np.empty((h_px, w_px), dtype=np.float32)
    chunk = 128
    for y0 in range(0, h_px, chunk):
        y1 = min(h_px, y0 + chunk)
        grid_y, grid_x = np.meshgrid(ys[y0:y1], xs, indexing="ij")
        pts = np.stack([grid_x.reshape(-1) / w_px, grid_y.reshape(-1) / h_px], axis=1)
        disp = field.predict_norm(pts).reshape(y1 - y0, w_px, 2)
        map_x[y0:y1] = grid_x - disp[:, :, 0] * w_px
        map_y[y0:y1] = grid_y - disp[:, :, 1] * h_px
    return cv2.remap(
        moving,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _ensure_moving_prerot(pair_id: int, matches: dict) -> tuple[np.ndarray, Path]:
    """IHC after pre-rotation only (no rigid) — moving image for direct field fit."""
    rd = run_dir(pair_id)
    rd.mkdir(parents=True, exist_ok=True)
    prerot_path = rd / "ihc_prerot.png"
    if prerot_path.exists():
        img = cv2.imread(str(prerot_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img, rd

    level = int(matches.get("preview_level") or DEFAULT_PREVIEW_LEVEL)
    pre_rot = float(matches.get("pre_rotation_deg") or 0.0)
    from crop_core import whole_gray

    ihc = whole_gray(pair_id, "ihc", level)
    if ihc is None:
        raise RuntimeError(f"no IHC preview for pair {pair_id} level {level}")
    ihc_prerot, _ = _rotate_gray(ihc, pre_rot)
    cv2.imwrite(str(prerot_path), ihc_prerot)
    return ihc_prerot, rd


def field_fit(
    pair_id: int,
    field_estimator: str = "tps",
    wendland_epsilon: float | None = None,
    bspline_grid: int | None = None,
    bspline_reg: float | None = None,
    inliers_only: bool = True,
    mode: str = "residual_after_rigid",
) -> dict:
    """
    mode:
      residual_after_rigid — anchors = he − rigid(ihc); warp ihc_rigid
      direct — anchors = he − ihc; warp ihc_prerot (no rigid)
    """
    from setup.coarse_to_fine.annotations import Anchor
    from setup.coarse_to_fine.field import fit_field

    if mode not in ("residual_after_rigid", "direct"):
        raise ValueError(f"unknown field mode {mode!r}")

    matches = _matches_for_fit(pair_id)
    rigid_prerot = matches.get("rigid_prerot")
    if rigid_prerot is None:
        run = load_run(pair_id)
        saved = load(pair_id)
        rigid_prerot = (run or {}).get("rigid_prerot") or (saved or {}).get("rigid_prerot")

    if mode == "residual_after_rigid" and rigid_prerot is None:
        raise RuntimeError("no rigid_prerot matrix for residual anchors")

    r00 = r01 = tx = r10 = r11 = ty = 0.0
    if rigid_prerot is not None:
        (r00, r01, tx), (r10, r11, ty) = rigid_prerot

    anchors: list[Anchor] = []
    overlay_src: list[dict] = []
    for p in matches.get("points") or []:
        hx, hy = float(p["he"][0]), float(p["he"][1])
        ix, iy = float(p["ihc"][0]), float(p["ihc"][1])
        rigid_x = r00 * ix + r01 * iy + tx
        rigid_y = r10 * ix + r11 * iy + ty
        inlier = bool(p.get("inlier", True))
        overlay_src.append(
            {
                "he": [hx, hy],
                "ihc": [ix, iy],
                "rigid": [rigid_x, rigid_y],
                "inlier": inlier,
            }
        )
        if inliers_only and not inlier:
            continue
        if mode == "direct":
            du, dv = hx - ix, hy - iy
        else:
            du, dv = hx - rigid_x, hy - rigid_y
        anchors.append(
            Anchor(
                px=hx,
                py=hy,
                du=du,
                dv=dv,
                weight=max(float(p.get("score") or 1.0), 1e-6),
            )
        )
    if len(anchors) < 3:
        raise RuntimeError(f"need >=3 anchors for field fit, got {len(anchors)}")

    field = fit_field(
        anchors,
        field_estimator=field_estimator,
        wendland_epsilon=wendland_epsilon,
        bspline_grid=bspline_grid,
        bspline_reg=bspline_reg,
    )

    if mode == "direct":
        moving, rd = _ensure_moving_prerot(pair_id, matches)
        he_path = rd / "he.png"
        if not he_path.exists():
            from crop_core import whole_gray

            level = int(matches.get("preview_level") or DEFAULT_PREVIEW_LEVEL)
            he = whole_gray(pair_id, "he", level)
            if he is not None:
                cv2.imwrite(str(he_path), he)
    else:
        _he, moving, rd = _ensure_rigid_preview(pair_id, matches)

    h_px, w_px = moving.shape[:2]
    warped = _warp_field_preview(moving, field)
    stem = "direct" if mode == "direct" else "residual"
    est = field.estimator
    preview_name = f"field_preview_{stem}_{est}.png"
    fit_name = f"field_fit_{stem}_{est}.json"
    cv2.imwrite(str(rd / preview_name), warped)
    cv2.imwrite(str(rd / f"field_preview_{stem}.png"), warped)
    if mode == "residual_after_rigid":
        cv2.imwrite(str(rd / "field_preview.png"), warped)

    pts_a = np.array([[a.px, a.py] for a in anchors], dtype=float)
    truth = np.array([[a.du, a.dv] for a in anchors], dtype=float)
    pred = field.predict_norm(pts_a)
    rmse = float(np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=1))))

    he_pts_n = np.array([o["he"] for o in overlay_src], dtype=float)
    disp_all = field.predict_norm(he_pts_n)
    overlay_corrs = []
    tre_sum = 0.0
    tre_sq = 0.0
    tre_n = 0
    for o, d in zip(overlay_src, disp_all):
        if mode == "direct":
            sx, sy = o["ihc"]
        else:
            sx, sy = o["rigid"]
        he_x, he_y = o["he"][0] * w_px, o["he"][1] * h_px
        field_x = (sx + float(d[0])) * w_px
        field_y = (sy + float(d[1])) * h_px
        overlay_corrs.append(
            {
                "he": [he_x, he_y],
                "rigid": [o["rigid"][0] * w_px, o["rigid"][1] * h_px],
                "source": [sx * w_px, sy * h_px],
                "field": [field_x, field_y],
                "inlier": o["inlier"],
            }
        )
        if o["inlier"]:
            dist = float(np.hypot(he_x - field_x, he_y - field_y))
            tre_sum += dist
            tre_sq += dist * dist
            tre_n += 1

    tre_mean_px = float(tre_sum / tre_n) if tre_n else float("nan")
    tre_rmse_px = float(np.sqrt(tre_sq / tre_n)) if tre_n else float("nan")

    payload = {
        "pair_id": pair_id,
        "mode": mode,
        "field_estimator": est,
        "wendland_epsilon": wendland_epsilon,
        "bspline_grid": bspline_grid,
        "bspline_reg": bspline_reg,
        "inliers_only": inliers_only,
        "n_anchors": len(anchors),
        "rmse_norm": rmse,
        "tre_mean_px": tre_mean_px,
        "tre_rmse_px": tre_rmse_px,
        "tre_n": tre_n,
        "kind": field.kind,
        "width": float(w_px),
        "height": float(h_px),
        "preview": preview_name,
        "overlay_corrs": overlay_corrs,
        "ran_at": int(time.time()),
    }
    text = json.dumps(payload, separators=(",", ":"))
    (rd / fit_name).write_text(text)
    (rd / f"field_fit_{stem}.json").write_text(text)
    (rd / "field_fit.json").write_text(text)
    return payload
