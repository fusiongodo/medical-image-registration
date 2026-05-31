"""
Shared tile loading and rendering utilities.

load_tile_gray(job, page_cache) -> np.ndarray (CNN_H, CNN_W) float32
render_tile_with_keypoints(ax, img, pts_xy, title, ...) -> None
render_tile_grid(jobs, page_cache, ncols, kp_color, kp_size) -> None
"""

import math

import tifffile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_tile_gray(job: dict, page_cache: dict | None = None) -> np.ndarray:
    """
    Load the fixed tile referenced by a keypoint-annotation job dict.

    Parameters
    ----------
    job : dict
        Entry from {OS}_he_keypoint_annotations_superpoint.json.
        Required keys: fixed_path, pyramid_page_idx, grid, x_idx, y_idx,
                       cnn_input_height, cnn_input_width.
    page_cache : dict | None
        Optional dict keyed by (fixed_path, pyramid_page_idx) to avoid
        re-reading the same TIFF page across multiple calls.

    Returns
    -------
    np.ndarray
        Shape (cnn_input_height, cnn_input_width), dtype float32 in [0, 1].
    """
    if page_cache is None:
        page_cache = {}

    key = (job["fixed_path"], int(job["pyramid_page_idx"]))
    if key not in page_cache:
        with tifffile.TiffFile(job["fixed_path"]) as slide:
            page_cache[key] = slide.pages[int(job["pyramid_page_idx"])].asarray()

    page = page_cache[key]
    H, W = page.shape[:2]

    grid = job["grid"]
    x_idx, y_idx = job["x_idx"], job["y_idx"]
    tile_w = W // grid
    tile_h = H // grid
    x0, y0 = x_idx * tile_w, y_idx * tile_h
    x1 = W if x_idx == grid - 1 else x0 + tile_w
    y1 = H if y_idx == grid - 1 else y0 + tile_h
    crop = page[y0:y1, x0:x1]

    pil = Image.fromarray(crop.astype(np.uint8) if crop.dtype != np.uint8 else crop).convert("L")
    pil = pil.resize((job["cnn_input_width"], job["cnn_input_height"]), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


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
    ncols: int = 5,
    kp_color: str = "lime",
    kp_size: int = 8,
    min_thresh: float | None = None,
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
    """
    if min_thresh is not None:
        jobs = [j for j in jobs if j.get("conf_thresh", 0.0) >= min_thresh]

    n = len(jobs)
    if n == 0:
        return

    if page_cache is None:
        page_cache = {}

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).reshape(-1)

    for i, job in enumerate(jobs):
        img = load_tile_gray(job, page_cache)
        pts_xy = job.get("fixed_keypoints_cnn")
        n_kp = len(pts_xy) if pts_xy else 0
        title = (
            f"depth={job['crop_depth']} "
            f"count={n_kp} "
            f"thresh={job['conf_thresh']:.3f}"
        )
        render_tile_with_keypoints(
            axes[i], img, pts_xy, title,
            kp_color=kp_color, kp_size=kp_size,
        )

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
