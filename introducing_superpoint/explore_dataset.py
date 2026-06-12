"""
Visualise StainPairKeypointDataset items: HE/IHC tensors + shared GT keypoints.

tensor_to_gray_numpy(image) -> (H, W) float32 [0, 1]
render_training_pair(item) -> Figure
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


def render_training_grid(dataset, indices=None, ncols=2, n_samples=8, seed=0):
    """
    dataset: StainPairKeypointDataset
    indices: optional list of dataset indices; if None, sample n_samples random indices
    returns: matplotlib Figure with HE row and IHC row per sample group
    """
    if indices is None:
        rng = np.random.default_rng(seed)
        n_samples = min(n_samples, len(dataset))
        indices = sorted(rng.choice(len(dataset), size=n_samples, replace=False).tolist())

    n = len(indices)
    if n == 0:
        raise ValueError("no indices to render")

    rows_per_group = 2
    n_groups = math.ceil(n / ncols)
    total_rows = n_groups * rows_per_group

    fig, axes = plt.subplots(total_rows, ncols, figsize=(ncols * 3.5, total_rows * 3))
    axes = np.array(axes).reshape(-1)
    used = set()

    for i, idx in enumerate(indices):
        item = dataset[idx]
        group = i // ncols
        col = i % ncols
        he_ax_idx = group * rows_per_group * ncols + col
        ihc_ax_idx = (group * rows_per_group + 1) * ncols + col

        img_he = tensor_to_gray_numpy(item["image_he"])
        img_ihc = tensor_to_gray_numpy(item["image_ihc"])
        pts = _gt_to_numpy(item["gt_keypoints"])
        meta = item["meta"]
        n_kp = 0 if pts is None else len(pts)

        render_tile_with_keypoints(
            axes[he_ax_idx], img_he, pts, _meta_title("HE", meta, n_kp),
            kp_color="lime", kp_size=10,
        )
        render_tile_with_keypoints(
            axes[ihc_ax_idx], img_ihc, pts, _meta_title("IHC", meta, n_kp),
            kp_color="lime", kp_size=10,
        )
        used.add(he_ax_idx)
        used.add(ihc_ax_idx)

    for ax_idx, ax in enumerate(axes):
        if ax_idx not in used:
            ax.axis("off")

    plt.tight_layout()
    return fig


def save_training_pair(item, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_training_pair(item)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
