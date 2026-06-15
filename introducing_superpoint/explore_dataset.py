"""
Visualise StainPairKeypointDataset items: HE/IHC tensors + shared GT keypoints.

tensor_to_gray_numpy(image) -> (H, W) float32 [0, 1]
render_training_pair(item) -> Figure
render_stain_pair_grid(dataset, indices, points_for_side, ...) -> Figure
render_training_grid(dataset, indices, ncols, n_samples, seed) -> Figure
save_training_pair(item, path) -> Path
"""
import argparse
import importlib
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
import conf
importlib.reload(conf)

from core.fig_io import show_or_save_figure
from core.tile_vis import render_tile_with_keypoints
from dataset import StainPairKeypointDataset


def tensor_to_gray_numpy(image: torch.Tensor) -> np.ndarray:
    """
    image: (1, H, W) or (H, W) float tensor in [0, 1]
    returns: (H, W) float32 numpy
    """
    if image.dim() == 3:
        image = image.squeeze(0)
    return image.detach().cpu().numpy().astype(np.float32)


def _gt_to_numpy(gt_keypoints: torch.Tensor):
    if gt_keypoints is None or gt_keypoints.numel() == 0:
        return None
    return gt_keypoints.detach().cpu().numpy()


def _meta_title(prefix: str, meta: dict, n_kp: int) -> str:
    return (
        f"{prefix} pair={meta['pair_id']} depth={meta['crop_depth']} "
        f"x={meta['x_idx']} y={meta['y_idx']} kp={n_kp}"
    )


def render_training_pair(item, figsize=(10, 4)):
    """
    item: dict from StainPairKeypointDataset.__getitem__
    returns: matplotlib Figure — left HE, right IHC, same GT on both
    """
    img_he = tensor_to_gray_numpy(item["image_he"])
    img_ihc = tensor_to_gray_numpy(item["image_ihc"])
    pts = _gt_to_numpy(item["gt_keypoints"])
    meta = item["meta"]
    n_kp = 0 if pts is None else len(pts)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    render_tile_with_keypoints(
        axes[0], img_he, pts, _meta_title("HE", meta, n_kp),
        kp_color="lime", kp_size=12,
    )
    render_tile_with_keypoints(
        axes[1], img_ihc, pts, _meta_title("IHC", meta, n_kp),
        kp_color="lime", kp_size=12,
    )
    plt.tight_layout()
    return fig


def render_stain_pair_grid(
    dataset,
    indices,
    points_for_side,
    ncols=2,
    kp_color="lime",
    kp_size=10,
):
    """
    dataset: StainPairKeypointDataset
    indices: list of dataset indices
    points_for_side: callable(item, "he"|"ihc") -> (N, 2+) ndarray or None
    ncols: number of HE/IHC pairs per row (each pair uses two adjacent columns)
    returns: matplotlib Figure with HE and IHC side-by-side per tile
    """
    n = len(indices)
    if n == 0:
        raise ValueError("no indices to render")

    pair_cols = ncols * 2
    total_rows = math.ceil(n / ncols)

    fig, axes = plt.subplots(total_rows, pair_cols, figsize=(pair_cols * 3.5, total_rows * 3))
    axes = np.array(axes, dtype=object).reshape(total_rows, pair_cols)
    used = set()

    for i, idx in enumerate(indices):
        item = dataset[idx]
        row = i // ncols
        pair_col = i % ncols

        img_he = tensor_to_gray_numpy(item["image_he"])
        img_ihc = tensor_to_gray_numpy(item["image_ihc"])
        meta = item["meta"]

        pts_he = points_for_side(item, "he")
        pts_ihc = points_for_side(item, "ihc")
        n_kp_he = 0 if pts_he is None else len(pts_he)
        n_kp_ihc = 0 if pts_ihc is None else len(pts_ihc)

        he_ax = axes[row, pair_col * 2]
        ihc_ax = axes[row, pair_col * 2 + 1]
        render_tile_with_keypoints(
            he_ax, img_he, pts_he, _meta_title("HE", meta, n_kp_he),
            kp_color=kp_color, kp_size=kp_size,
        )
        render_tile_with_keypoints(
            ihc_ax, img_ihc, pts_ihc, _meta_title("IHC", meta, n_kp_ihc),
            kp_color=kp_color, kp_size=kp_size,
        )
        used.add((row, pair_col * 2))
        used.add((row, pair_col * 2 + 1))

    for row in range(total_rows):
        for col in range(pair_cols):
            if (row, col) not in used:
                axes[row, col].axis("off")

    plt.tight_layout()
    return fig


def render_training_grid(dataset, indices=None, ncols=2, n_samples=8, seed=0):
    """
    dataset: StainPairKeypointDataset
    indices: optional list of dataset indices; if None, sample n_samples random indices
    returns: matplotlib Figure with HE and IHC side-by-side per tile
    """
    if indices is None:
        rng = np.random.default_rng(seed)
        n_samples = min(n_samples, len(dataset))
        indices = sorted(rng.choice(len(dataset), size=n_samples, replace=False).tolist())

    def _gt_for_side(item, side):
        return _gt_to_numpy(item["gt_keypoints"])

    return render_stain_pair_grid(dataset, indices, _gt_for_side, ncols=ncols)


def save_training_pair(item, path: Path) -> Path:
    path = Path(path)
    fig = render_training_pair(item)
    show_or_save_figure(fig, path)
    return path


def find_dataset_index(dataset, pair_id, crop_depth, x_idx, y_idx):
    for idx, job in enumerate(dataset.tile_jobs):
        if (
            job["pair_id"] == pair_id
            and job["crop_depth"] == crop_depth
            and job["x_idx"] == x_idx
            and job["y_idx"] == y_idx
        ):
            return idx
    raise ValueError(
        f"no job for pair_id={pair_id} depth={crop_depth} x={x_idx} y={y_idx}"
    )


def main():
    parser = argparse.ArgumentParser(description="Explore StainPairKeypointDataset tiles")
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--random", type=int, default=None, metavar="N")
    parser.add_argument("--pair-id", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    dataset = StainPairKeypointDataset()

    if args.pair_id is not None:
        if args.depth is None or args.x is None or args.y is None:
            parser.error("--pair-id requires --depth, --x, and --y")
        idx = find_dataset_index(dataset, args.pair_id, args.depth, args.x, args.y)
        item = dataset[idx]
        fig = render_training_pair(item)
    elif args.idx is not None:
        item = dataset[args.idx]
        fig = render_training_pair(item)
    elif args.random is not None:
        fig = render_training_grid(
            dataset, ncols=args.ncols, n_samples=args.random, seed=args.seed,
        )
    else:
        item = dataset[0]
        fig = render_training_pair(item)

    show_or_save_figure(fig, args.save)


if __name__ == "__main__":
    main()
