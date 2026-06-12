"""
Shared tile loading and rendering utilities.

is_background_tile(job) -> bool
load_tile_gray(job, page_cache, side) -> np.ndarray (CNN_H, CNN_W) float32
render_tile_with_keypoints(ax, img, pts_xy, title, ...) -> None
render_tile_grid(jobs, page_cache, ncols, kp_color, kp_size, min_thresh, show_moving, exclude_background) -> None
"""

import math
import sys
from pathlib import Path

import tifffile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import conf


def _load_page_crop_gray(
    path: str,
    pyramid_page_idx: int,
    grid: int,
    x_idx: int,
    y_idx: int,
    cnn_input_height: int,
    cnn_input_width: int,
    page_cache: dict,
) -> np.ndarray:
    path = conf.resolve(path)
    key = (str(path), pyramid_page_idx)
    if key not in page_cache:
        with tifffile.TiffFile(path) as slide:
            page_cache[key] = slide.pages[pyramid_page_idx].asarray()

    page = page_cache[key]
    H, W = page.shape[:2]
    tile_w = W // grid
    tile_h = H // grid
    x0, y0 = x_idx * tile_w, y_idx * tile_h
    x1 = W if x_idx == grid - 1 else x0 + tile_w
    y1 = H if y_idx == grid - 1 else y0 + tile_h
    crop = page[y0:y1, x0:x1]

    pil = Image.fromarray(crop.astype(np.uint8) if crop.dtype != np.uint8 else crop).convert("L")
    pil = pil.resize((cnn_input_width, cnn_input_height), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def is_background_tile(job: dict) -> bool:
    """Return the precomputed binary_mask_excluded flag from the job dict."""
    return bool(job.get("binary_mask_excluded", False))


def load_tile_gray(
    job: dict,
    page_cache: dict | None = None,
    side: str = "fixed",
) -> np.ndarray:
    """
    Load a tile from a keypoint-annotation job dict.

    Parameters
    ----------
    job : dict
        Entry from {OS}_he_keypoint_annotations_superpoint.json.
    page_cache : dict | None
        Optional dict keyed by (path, pyramid_page_idx) to avoid
        re-reading the same TIFF page across calls.
    side : str
        "fixed" or "moving" — which image of the pair to load.

    Returns
    -------
    np.ndarray
        Shape (cnn_input_height, cnn_input_width), dtype float32 in [0, 1].
    """
    if page_cache is None:
        page_cache = {}

    path = conf.job_image_path(job, side)
    return _load_page_crop_gray(
        str(conf.to_relative(path)),
        int(job["pyramid_page_idx"]),
        job["grid"],
        job["x_idx"],
        job["y_idx"],
        job["cnn_input_height"],
        job["cnn_input_width"],
        page_cache,
    )


def render_tile_with_keypoints(
    ax,
    img: np.ndarray,
    pts_xy,
    title: str,
    cmap: str | None = "gray",
    kp_color: str = "lime",
    kp_size: int = 8,
    fontsize: int | None = None,
) -> None:
    """
    Render a single tile with optional keypoint overlay.

    Parameters
    ----------
    ax : matplotlib Axes
    img : np.ndarray
        (H, W) float32 grayscale or (H, W, 3) RGB.
    pts_xy : array-like of shape (N, 2+) or None
        Each row is [x, y, ...]. Pass None for no keypoints.
    title : str
    cmap : str | None
        Passed to imshow; use None for RGB images.
    kp_color : str
    kp_size : int
        Marker area in points² passed to scatter.
    fontsize : int | None
        Title font size; uses matplotlib default when None.
    """
    title_kwargs = {} if fontsize is None else {"fontsize": fontsize}
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_title(title, **title_kwargs)
    ax.axis("off")
    if pts_xy is not None and len(pts_xy) > 0:
        pts = np.asarray(pts_xy)
        ax.scatter(pts[:, 0], pts[:, 1], s=kp_size, color=kp_color, linewidths=0)


def render_tile_grid(
    jobs: list[dict],
    page_cache: dict | None = None,
    ncols: int = 2,
    kp_color: str = "lime",
    kp_size: int = 8,
    min_thresh: float | None = None,
    show_moving: bool = False,
    exclude_background: bool | None = None,
) -> None:
    """
    Render a grid of tiles with their stored fixed_keypoints_cnn overlaid.

    Parameters
    ----------
    jobs : list[dict]
        Subset of keypoint-annotation job dicts to visualise.
    page_cache : dict | None
        Shared page cache passed to load_tile_gray.
    ncols : int
        Number of columns in the grid.
    kp_color : str
    kp_size : int
    min_thresh : float | None
        When set, only jobs whose conf_thresh >= min_thresh are rendered.
    exclude_background : bool | None
        None  — show all tiles (default).
        True  — show only tissue tiles (binary_mask_excluded == False).
        False — show only background tiles (binary_mask_excluded == True).
    show_moving : bool
        When True, renders the moving tile in the row directly below each
        fixed-tile row.  Layout for ncols=4, n=7:
            Row 0 (fixed):  job0 job1 job2 job3 job4
            Row 1 (moving): job0 job1 job2 job3 job4
            Row 2 (fixed):  job5 job6  —    —    —
            Row 3 (moving): job5 job6  —    —    —
    """
    if min_thresh is not None:
        jobs = [j for j in jobs if j.get("conf_thresh", 0.0) >= min_thresh]

    if exclude_background is True:
        jobs = [j for j in jobs if not is_background_tile(j)]
    elif exclude_background is False:
        jobs = [j for j in jobs if is_background_tile(j)]

    n = len(jobs)
    if n == 0:
        return

    if page_cache is None:
        page_cache = {}

    rows_per_group = 2 if show_moving else 1
    n_groups = math.ceil(n / ncols)
    total_rows = n_groups * rows_per_group

    fig, axes = plt.subplots(total_rows, ncols, figsize=(ncols * 3, total_rows * 3))
    axes = np.array(axes).reshape(-1)

    used = set()
    for i, job in enumerate(jobs):
        group = i // ncols
        col = i % ncols
        fixed_ax_idx = group * rows_per_group * ncols + col

        pts_xy = job.get("fixed_keypoints_cnn")
        n_kp = len(pts_xy) if pts_xy else 0
        thresh = job.get("conf_thresh")
        thresh_str = f"  thresh={thresh:.3f}" if thresh is not None else ""
        fixed_title = f"f: pair_id={job['pair_id']} depth={job['crop_depth']} x_idx={job['x_idx']} y_idx={job['y_idx']}"
        #fixed_title = f"fixed  depth={job['crop_depth']}  {n_kp} kp{thresh_str}  pair_id={job['pair_id']} x_idx={job['x_idx']} y_idx={job['y_idx']}"

        render_tile_with_keypoints(
            axes[fixed_ax_idx],
            load_tile_gray(job, page_cache, side="fixed"),
            pts_xy, fixed_title,
            kp_color=kp_color, kp_size=kp_size,
        )
        used.add(fixed_ax_idx)

        if show_moving:
            moving_ax_idx = (group * rows_per_group + 1) * ncols + col
            render_tile_with_keypoints(
                axes[moving_ax_idx],
                load_tile_gray(job, page_cache, side="moving"),
                None,
                f"m: depth={job['crop_depth']} pair_id={job['pair_id']} x_idx={job['x_idx']} y_idx={job['y_idx']}",
                kp_color=kp_color, kp_size=kp_size,
            )
            used.add(moving_ax_idx)

    for idx, ax in enumerate(axes):
        if idx not in used:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
