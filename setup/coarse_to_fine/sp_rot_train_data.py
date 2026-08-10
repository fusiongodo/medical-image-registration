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


def _scan_tiles(pairs: list[int], depth: int) -> list[tuple[int, int, int, str]]:
    """Prefer live tissue_tiles; fall back to data/cropped index."""
    import crop_core

    out: list[tuple[int, int, int, str]] = []
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
            out.append((pid, x, y, loc))
    return out


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
    """Map coords in base OUT crop → warped OUT crop (both size `out`)."""
    off = (src - out) / 2.0
    T_out_to_src = np.array(
        [[1, 0, off], [0, 1, off], [0, 0, 1]], dtype=np.float32
    )
    T_src_to_out = np.array(
        [[1, 0, -off], [0, 1, -off], [0, 0, 1]], dtype=np.float32
    )
    H_rot = _affine_to_homography(M_src)
    return T_src_to_out @ H_rot @ T_out_to_src


class RotWarpDataset(Dataset):
    def __init__(
        self,
        pairs: list[int] | None = None,
        depth: int = DEFAULT_DEPTH,
        preview_level: int = DEFAULT_PREVIEW_LEVEL,
        src_size: int = SRC_SIZE,
        out_size: int = OUT_SIZE,
        sides: tuple[str, ...] = ("he", "ihc"),
    ):
        self.pairs = list(pairs or DEFAULT_PAIRS)
        self.depth = int(depth)
        self.preview_level = int(preview_level)
        self.src_size = int(src_size)
        self.out_size = int(out_size)
        self.sides = sides
        self.tiles = _scan_tiles(self.pairs, self.depth)
        if not self.tiles:
            raise RuntimeError(
                f"no unmasked tiles for pairs={self.pairs} depth={self.depth}"
            )
        self._page_cache: dict[tuple[int, str], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.tiles)

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
        pair_id, tx, ty, _loc = self.tiles[idx]
        side = random.choice(self.sides)
        page = self._page(pair_id, side)
        scale = 2 ** (self.depth - self.preview_level)
        cx = (tx + 0.5) * conf.CNN_INPUT_WIDTH / scale
        cy = (ty + 0.5) * conf.CNN_INPUT_HEIGHT / scale
        src = _crop_patch(page, cx, cy, self.src_size)
        theta = random.uniform(-180.0, 180.0)
        warped_src, M, valid_src = _rotate_center(src, theta)
        base = _center_square(src, self.out_size)
        warped = _center_square(warped_src, self.out_size)
        valid = _center_square(valid_src, self.out_size)
        H = _compose_crop_homography(M, self.src_size, self.out_size)
        base_t = torch.from_numpy(base.astype(np.float32) / 255.0).unsqueeze(0)
        warp_t = torch.from_numpy(warped.astype(np.float32) / 255.0).unsqueeze(0)
        valid_t = torch.from_numpy((valid > 127).astype(np.float32))
        return {
            "image": base_t,
            "warped": warp_t,
            "valid_mask": valid_t,
            "homography": torch.from_numpy(H),
            "theta_deg": float(theta),
            "pair_id": int(pair_id),
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
