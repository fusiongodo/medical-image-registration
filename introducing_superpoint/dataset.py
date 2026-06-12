"""
Matching HE/IHC tile pairs with HE pseudo-GT keypoints for SuperPoint retraining.

StainPairKeypointDataset[i] -> {
    "image_he":     [1, CNN_H, CNN_W] float32 in [0, 1]   (fixed / H&E)
    "image_ihc":    [1, CNN_H, CNN_W] float32 in [0, 1]   (moving / IHC, HE frame)
    "gt_keypoints": [N, 3] float32 — (x, y, conf) in HE CNN pixels
    "meta":         dict
}
make_loader(...) -> DataLoader; collate keeps gt_keypoints as list[Tensor].
"""
import sys
import json
import importlib
from pathlib import Path

import tifffile
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

sys.path.append(str(Path(__file__).resolve().parent.parent))
import conf
importlib.reload(conf)


class StainPairKeypointDataset(Dataset):
    def __init__(
        self,
        annotation_path=None,
        input_height=None,
        input_width=None,
        require_converged=True,
        exclude_background=True,
    ):
        self.annotation_path = Path(annotation_path or conf.HE_KEYPOINT_ANNOTATION_PATH)
        self.input_height = input_height or conf.CNN_INPUT_HEIGHT
        self.input_width = input_width or conf.CNN_INPUT_WIDTH

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            jobs = json.load(f)

        self.tile_jobs = [
            job for job in jobs
            if (not require_converged or job.get("converged", False))
            and (not exclude_background or not job.get("binary_mask_excluded", False))
        ]

        self._page_cache = {}

    def __len__(self):
        return len(self.tile_jobs)

    def _load_page_array(self, path, pyramid_page_idx):
        path = conf.resolve(path)
        key = (str(path), int(pyramid_page_idx))
        if key not in self._page_cache:
            with tifffile.TiffFile(path) as slide:
                self._page_cache[key] = slide.pages[pyramid_page_idx].asarray()
        return self._page_cache[key]

    def _crop_tile(self, img, x_idx, y_idx, grid):
        H, W = img.shape[:2]
        tile_w = W // grid
        tile_h = H // grid
        x0 = x_idx * tile_w
        y0 = y_idx * tile_h
        x1 = W if x_idx == grid - 1 else x0 + tile_w
        y1 = H if y_idx == grid - 1 else y0 + tile_h
        return img[y0:y1, x0:x1]

    def _tile_to_tensor(self, tile):
        if tile.dtype != np.uint8:
            tile = tile.astype(np.uint8)
        image = Image.fromarray(tile).convert("L")
        image = image.resize(
            (self.input_width, self.input_height),
            resample=Image.BILINEAR,
        )
        return TF.to_tensor(image)

    def _load_side(self, job, side):
        path = conf.job_image_path(job, side)
        page = self._load_page_array(path, job["pyramid_page_idx"])
        tile = self._crop_tile(page, job["x_idx"], job["y_idx"], job["grid"])
        return self._tile_to_tensor(tile)

    def __getitem__(self, idx):
        job = self.tile_jobs[idx]

        image_he = self._load_side(job, "fixed")
        image_ihc = self._load_side(job, "moving")

        keypoints = job.get("fixed_keypoints_cnn") or []
        gt_keypoints = torch.tensor(keypoints, dtype=torch.float32).reshape(-1, 3)

        meta = {
            "pair_id": job["pair_id"],
            "source_image_id": job["source_image_id"],
            "target_image_id": job["target_image_id"],
            "crop_depth": job["crop_depth"],
            "grid": job["grid"],
            "x_idx": job["x_idx"],
            "y_idx": job["y_idx"],
            "pyramid_page_idx": job["pyramid_page_idx"],
        }

        return {
            "image_he": image_he,
            "image_ihc": image_ihc,
            "gt_keypoints": gt_keypoints,
            "meta": meta,
        }


def collate_pairs(batch):
    return {
        "image_he": torch.stack([item["image_he"] for item in batch]),
        "image_ihc": torch.stack([item["image_ihc"] for item in batch]),
        "gt_keypoints": [item["gt_keypoints"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def make_loader(
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    dataset=None,
    generator=None,
    **dataset_kwargs,
):
    dataset = dataset or StainPairKeypointDataset(**dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_pairs,
        generator=generator,
    )
