"""
Fits a smooth per-pair translation field from depth-5 phase-correlation displacements.

Outlier rejection:
  For each valid tile, compute the Frobenius norm of the displacement Jacobian
  (sparse finite differences).  Tiles whose norm exceeds mu + 2*sigma are dropped.

Smoothing:
  scipy RBFInterpolator with thin_plate_spline kernel, fitted on inlier tile centres
  in normalised [0,1]^2 space.  Evaluated at every tile centre for depths 0-5 and
  stored in tile-pixel units (344x512 space).

Usage:
  python setup/smooth_field.py [--pair-ids N ...] [--force]

Output:
  data/smooth/{pair_id}_smooth_field.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import conf

CROPPED_DIR = conf.PROJECT_ROOT / "data" / "cropped"
SMOOTH_DIR  = conf.PROJECT_ROOT / "data" / "smooth"
CNN_W       = conf.CNN_INPUT_WIDTH   # 512
CNN_H       = conf.CNN_INPUT_HEIGHT  # 344
FIT_DEPTH   = 5
FIT_GRID    = 2 ** FIT_DEPTH  # 32
MAX_DEPTH   = conf.MAX_CROP_DEPTH


def _load_grid(pair_id: int):
    """Return (dx_grid, dy_grid) each shape (32,32), NaN where missing."""
    dx_grid = np.full((FIT_GRID, FIT_GRID), np.nan)
    dy_grid = np.full((FIT_GRID, FIT_GRID), np.nan)
    depth_dir = CROPPED_DIR / str(pair_id) / f"d{FIT_DEPTH}"
    if not depth_dir.exists():
        return dx_grid, dy_grid
    for tile_dir in depth_dir.iterdir():
        if not tile_dir.is_dir():
            continue
        parts = tile_dir.name.split("_")
        if len(parts) != 2:
            continue
        try:
            xi, yi = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        disp_file = tile_dir / "elastix" / "displacement.json"
        if not disp_file.exists():
            continue
        d = json.loads(disp_file.read_text())
        dx_grid[yi, xi] = d["dx"]
        dy_grid[yi, xi] = d["dy"]
    return dx_grid, dy_grid


def _sparse_jacobian_norms(dx_grid, dy_grid):
    """
    Returns an array of (yi, xi, norm) for every valid tile.
    Norm = Frobenius norm of the 2x2 displacement Jacobian estimated via
    sparse central/one-sided finite differences over the tile grid.
    """
    valid = ~(np.isnan(dx_grid) | np.isnan(dy_grid))
    rows = []
    for yi in range(FIT_GRID):
        for xi in range(FIT_GRID):
            if not valid[yi, xi]:
                continue

            def _diff(grid, axis, i, j):
                if axis == 0:  # ∂/∂y (row direction)
                    hi = (i + 1 < FIT_GRID) and valid[i + 1, j]
                    lo = (i - 1 >= 0)        and valid[i - 1, j]
                    if hi and lo:
                        return (grid[i + 1, j] - grid[i - 1, j]) / 2.0
                    if hi:
                        return grid[i + 1, j] - grid[i, j]
                    if lo:
                        return grid[i, j] - grid[i - 1, j]
                    return 0.0
                else:  # ∂/∂x (col direction)
                    hi = (j + 1 < FIT_GRID) and valid[i, j + 1]
                    lo = (j - 1 >= 0)        and valid[i, j - 1]
                    if hi and lo:
                        return (grid[i, j + 1] - grid[i, j - 1]) / 2.0
                    if hi:
                        return grid[i, j + 1] - grid[i, j]
                    if lo:
                        return grid[i, j] - grid[i, j - 1]
                    return 0.0

            ddx_dy = _diff(dx_grid, 0, yi, xi)
            ddx_dx = _diff(dx_grid, 1, yi, xi)
            ddy_dy = _diff(dy_grid, 0, yi, xi)
            ddy_dx = _diff(dy_grid, 1, yi, xi)
            norm = math.sqrt(ddx_dy**2 + ddx_dx**2 + ddy_dy**2 + ddy_dx**2)
            rows.append((yi, xi, norm))
    return rows


def _fit_pair(pair_id: int, force: bool) -> bool:
    out_path = SMOOTH_DIR / f"{pair_id}_smooth_field.json"
    if out_path.exists() and not force:
        print(f"pair {pair_id}: skip (already computed, use --force to recompute)")
        return False

    dx_grid, dy_grid = _load_grid(pair_id)
    valid_count = int(np.sum(~np.isnan(dx_grid)))
    if valid_count < 4:
        print(f"pair {pair_id}: only {valid_count} displacement tiles at d{FIT_DEPTH}, skipping")
        return False

    jac_rows = _sparse_jacobian_norms(dx_grid, dy_grid)
    norms = np.array([r[2] for r in jac_rows])
    mu, sigma = norms.mean(), norms.std()
    threshold = mu + 2.0 * sigma

    outlier_set = {(r[0], r[1]) for r in jac_rows if r[2] > threshold}
    inlier_list = [
        (yi, xi)
        for yi in range(FIT_GRID)
        for xi in range(FIT_GRID)
        if not np.isnan(dx_grid[yi, xi]) and (yi, xi) not in outlier_set
    ]

    n_in  = len(inlier_list)
    n_out = len(outlier_set)

    if n_in < 4:
        print(f"pair {pair_id}: too few inliers ({n_in}), skipping")
        return False

    cx_in = np.array([(xi + 0.5) / FIT_GRID for yi, xi in inlier_list])
    cy_in = np.array([(yi + 0.5) / FIT_GRID for yi, xi in inlier_list])
    pts_in = np.stack([cx_in, cy_in], axis=1)

    # normalise displacements to full-image fraction
    dx_norm = np.array([dx_grid[yi, xi] / (FIT_GRID * CNN_W) for yi, xi in inlier_list])
    dy_norm = np.array([dy_grid[yi, xi] / (FIT_GRID * CNN_H) for yi, xi in inlier_list])

    rbf_dx = RBFInterpolator(pts_in, dx_norm, kernel="thin_plate_spline", smoothing=1e-3)
    rbf_dy = RBFInterpolator(pts_in, dy_norm, kernel="thin_plate_spline", smoothing=1e-3)

    depths_out: dict[str, dict[str, dict[str, float]]] = {}
    for depth in range(MAX_DEPTH + 1):
        g = 2 ** depth
        tiles: dict[str, dict[str, float]] = {}
        query_pts = []
        tile_names = []
        for yi in range(g):
            for xi in range(g):
                query_pts.append([(xi + 0.5) / g, (yi + 0.5) / g])
                tile_names.append(f"{xi}_{yi}")
        query_pts_arr = np.array(query_pts)
        dx_q = rbf_dx(query_pts_arr)
        dy_q = rbf_dy(query_pts_arr)
        for name, dxv, dyv in zip(tile_names, dx_q, dy_q):
            tiles[name] = {
                "dx": float(dxv * g * CNN_W),
                "dy": float(dyv * g * CNN_H),
            }
        depths_out[str(depth)] = tiles

    outlier_tiles = [f"{xi}_{yi}" for yi, xi in sorted(outlier_set)]
    result = {
        "pair_id":      pair_id,
        "fit_depth":    FIT_DEPTH,
        "n_inliers":    n_in,
        "n_outliers":   n_out,
        "outlier_tiles": outlier_tiles,
        "depths":       depths_out,
    }
    SMOOTH_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, separators=(",", ":")))
    print(f"pair {pair_id}: {n_in} inliers, {n_out} outliers → saved {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-ids", type=int, nargs="+")
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args()

    if args.pair_ids:
        pair_ids = args.pair_ids
    else:
        pair_ids = sorted(
            int(p.name) for p in CROPPED_DIR.iterdir()
            if p.is_dir() and p.name.isdigit()
        )

    for pid in pair_ids:
        _fit_pair(pid, args.force)


if __name__ == "__main__":
    main()
