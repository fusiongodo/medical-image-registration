"""Self-warp rotation dataset for SP rot-inv fine-tuning."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))

import conf
from setup.coarse_to_fine import masks as c2f_masks

DEFAULT_PAIRS = [0, 1, 3, 16]
DEFAULT_DEPTH = 5
DEFAULT_PREVIEW_LEVEL = 2
SRC_SIZE = 768
OUT_SIZE = 512
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_SPLIT_SEED = 0


def scan_tiles(pairs: list[int], depth: int) -> list[dict]:
    """Prefer live tissue_tiles; fall back to data/cropped index."""
    import crop_core

    out: list[dict] = []
    for pid in pairs:
        entries = c2f_masks.load(pid)
        try:
            info = crop_core.tissue_tiles(pid, depth)
            locs = list(info.get("tiles") or [])
        except Exception:
            locs = []
        if not locs:
            depth_dir = conf.PROJECT_ROOT / "data" / "cropped" / str(pid) / f"d{depth}"
            if depth_dir.is_dir():
                locs = [p.name for p in sorted(depth_dir.iterdir()) if p.is_dir()]
        for loc in locs:
            if entries and c2f_masks.is_masked(entries, depth, loc):
                continue
            try:
                x_s, y_s = loc.split("_")
                x, y = int(x_s), int(y_s)
            except ValueError:
                continue
            out.append({"pair_id": int(pid), "x": x, "y": y, "loc": loc})
    return out


def split_tiles(
    tiles: list[dict],
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    if not tiles:
        raise RuntimeError("no tiles to split")
    r_train, r_val, r_test = ratios
    if abs((r_train + r_val + r_test) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    order = list(range(len(tiles)))
    rng = random.Random(int(seed))
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1) if n > 1 else 0
        n_train = min(n_train, n - n_val - (1 if n > n_train + n_val else 0))
    n_test = n - n_train - n_val
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        n_test = n - n_train - n_val
    i0 = n_train
    i1 = n_train + n_val
    train_idx = order[:i0]
    val_idx = order[i0:i1]
    test_idx = order[i1:]
    return {
        "seed": int(seed),
        "ratios": [float(r_train), float(r_val), float(r_test)],
        "n_total": n,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train": [tiles[i] for i in train_idx],
        "val": [tiles[i] for i in val_idx],
        "test": [tiles[i] for i in test_idx],
    }


def subsample_tiles(tiles: list[dict], max_tiles: int | None, seed: int) -> list[dict]:
    if max_tiles is None or max_tiles <= 0 or len(tiles) <= max_tiles:
        return list(tiles)
    order = list(range(len(tiles)))
    rng = random.Random(int(seed))
    rng.shuffle(order)
    return [tiles[i] for i in order[: int(max_tiles)]]


def _crop_patch(page: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = page.shape[:2]
    half = size / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1, y1 = x0 + size, y0 + size
    out = np.full((size, size), 255, dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    if sx1 > sx0 and sy1 > sy0:
        out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = page[sy0:sy1, sx0:sx1]
    return out


def _rotate_center(img: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    warped = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    valid = cv2.warpAffine(
        np.ones((h, w), dtype=np.uint8) * 255,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, M, valid


def _center_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return img[y0 : y0 + size, x0 : x0 + size]


def _affine_to_homography(M: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M.astype(np.float32)
    return H


def _compose_crop_homography(M_src: np.ndarray, src: int, out: int) -> np.ndarray:
    off = (src - out) / 2.0
    T_out_to_src = np.array([[1, 0, off], [0, 1, off], [0, 0, 1]], dtype=np.float32)
    T_src_to_out = np.array([[1, 0, -off], [0, 1, -off], [0, 0, 1]], dtype=np.float32)
    H_rot = _affine_to_homography(M_src)
    return T_src_to_out @ H_rot @ T_out_to_src


def make_warp_pair(
    page: np.ndarray,
    tile: dict,
    *,
    depth: int,
    preview_level: int,
    src_size: int,
    out_size: int,
    theta_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = 2 ** (depth - preview_level)
    cx = (tile["x"] + 0.5) * conf.CNN_INPUT_WIDTH / scale
    cy = (tile["y"] + 0.5) * conf.CNN_INPUT_HEIGHT / scale
    src = _crop_patch(page, cx, cy, src_size)
    warped_src, M, valid_src = _rotate_center(src, theta_deg)
    base = _center_square(src, out_size)
    warped = _center_square(warped_src, out_size)
    valid = _center_square(valid_src, out_size)
    H = _compose_crop_homography(M, src_size, out_size)
    return base, warped, valid, H


class RotWarpDataset(Dataset):
    def __init__(
        self,
        tiles: list[dict],
        *,
        depth: int = DEFAULT_DEPTH,
        preview_level: int = DEFAULT_PREVIEW_LEVEL,
        src_size: int = SRC_SIZE,
        out_size: int = OUT_SIZE,
        sides: tuple[str, ...] = ("he", "ihc"),
        fixed_angles: list[float] | None = None,
    ):
        self.tiles = list(tiles)
        self.depth = int(depth)
        self.preview_level = int(preview_level)
        self.src_size = int(src_size)
        self.out_size = int(out_size)
        self.sides = sides
        self.fixed_angles = fixed_angles
        if not self.tiles:
            raise RuntimeError("empty tile list")
        self._page_cache: dict[tuple[int, str], np.ndarray] = {}

    def __len__(self) -> int:
        if self.fixed_angles is None:
            return len(self.tiles)
        return len(self.tiles) * len(self.fixed_angles)

    def _page(self, pair_id: int, side: str) -> np.ndarray:
        key = (pair_id, side)
        if key not in self._page_cache:
            import crop_core

            g = crop_core.whole_gray(pair_id, side, self.preview_level)
            if g is None:
                raise RuntimeError(f"no whole_gray pair={pair_id} side={side}")
            self._page_cache[key] = g
        return self._page_cache[key]

    def __getitem__(self, idx: int) -> dict:
        if self.fixed_angles is None:
            tile = self.tiles[idx]
            theta = random.uniform(-180.0, 180.0)
        else:
            n_ang = len(self.fixed_angles)
            tile = self.tiles[idx // n_ang]
            theta = float(self.fixed_angles[idx % n_ang])
        side = random.choice(self.sides)
        page = self._page(int(tile["pair_id"]), side)
        base, warped, valid, H = make_warp_pair(
            page,
            tile,
            depth=self.depth,
            preview_level=self.preview_level,
            src_size=self.src_size,
            out_size=self.out_size,
            theta_deg=theta,
        )
        base_t = torch.from_numpy(base.astype(np.float32) / 255.0).unsqueeze(0)
        warp_t = torch.from_numpy(warped.astype(np.float32) / 255.0).unsqueeze(0)
        valid_t = torch.from_numpy((valid > 127).astype(np.float32))
        return {
            "image": base_t,
            "warped": warp_t,
            "valid_mask": valid_t,
            "homography": torch.from_numpy(H),
            "theta_deg": float(theta),
            "pair_id": int(tile["pair_id"]),
            "loc": tile.get("loc"),
            "side": side,
        }


def collate_rot(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "warped": torch.stack([b["warped"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "homography": torch.stack([b["homography"] for b in batch], dim=0),
        "theta_deg": [b["theta_deg"] for b in batch],
        "pair_id": [b["pair_id"] for b in batch],
        "side": [b["side"] for b in batch],
    }
