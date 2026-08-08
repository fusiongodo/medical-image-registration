"""Extract a normalised IHC→HE rigid from regWSI's initial displacement field."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import conf
from setup import datasets
from setup.coarse_to_fine.reg_branches import clear_lam_caches
from setup.coarse_to_fine.rigid_sp_lg import fit_rigid_kabsch


def _norm_from_pixel_rigid(R_px: np.ndarray, t_px: np.ndarray, w: int, h: int) -> np.ndarray:
    sx, sy = float(w), float(h)
    r00, r01 = float(R_px[0, 0]), float(R_px[0, 1])
    r10, r11 = float(R_px[1, 0]), float(R_px[1, 1])
    tx, ty = float(t_px[0]), float(t_px[1])
    return np.array(
        [
            [r00, r01 * (sy / sx), tx / sx],
            [r10 * (sx / sy), r11, ty / sy],
        ],
        dtype=float,
    )


def rigid_from_displacement_field(
    df_path: Path,
    *,
    sample_step: int = 64,
    inlier_px: float = 8.0,
) -> tuple[np.ndarray, dict]:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(df_path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 2:
        dx = arr[..., 0]
        dy = arr[..., 1]
    elif arr.shape[0] == 2:
        dx, dy = arr[0], arr[1]
    else:
        raise ValueError(f"unexpected DF shape {arr.shape} in {df_path}")

    h, w = dx.shape
    ys = np.arange(sample_step // 2, h, sample_step)
    xs = np.arange(sample_step // 2, w, sample_step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ihc = np.stack([grid_x.ravel().astype(float), grid_y.ravel().astype(float)], axis=1)
    he = np.stack(
        [
            ihc[:, 0] + dx[grid_y.ravel(), grid_x.ravel()],
            ihc[:, 1] + dy[grid_y.ravel(), grid_x.ravel()],
        ],
        axis=1,
    )
    R, t, mask, stats = fit_rigid_kabsch(he, ihc, inlier_px=inlier_px)
    rigid_n = _norm_from_pixel_rigid(R, t, w, h)
    out_stats = {
        "width": int(w),
        "height": int(h),
        "n_samples": int(len(ihc)),
    }
    if stats:
        out_stats.update(stats)
    return rigid_n, out_stats


def find_initial_df(pair_tmp: Path, out_dir: Path) -> Path | None:
    candidates = [
        pair_tmp / "Initial_Registration" / "displacement_field.mha",
        out_dir / "Initial_Registration" / "displacement_field.mha",
    ]
    for root in (pair_tmp, out_dir):
        if not root.is_dir():
            continue
        for p in root.rglob("displacement_field.mha"):
            if "Initial_Registration" in p.parts:
                candidates.append(p)
    for p in candidates:
        if p.is_file():
            return p
    return None


def persist_regwsi_rigid(
    pair_id: int,
    df_path: Path,
    *,
    dataset: str | None = None,
    source: str = "regwsi_initial",
) -> dict:
    ds = datasets.normalize_dataset(dataset) if dataset else datasets.active_dataset()
    rigid_n, stats = rigid_from_displacement_field(df_path)
    store = {
        "pair_id": int(pair_id),
        "dataset": ds,
        "version": "regwsi_initial",
        "identity": datasets.pair_fingerprint(pair_id, ds),
        "rigid": rigid_n.tolist(),
        "stats": stats,
        "source": source,
        "df_path": str(df_path),
        "saved_at": int(time.time()),
    }
    path = datasets.rigid_path(pair_id, ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))
    clear_lam_caches(pair_id, dataset=ds)
    return store
