from pathlib import Path
import json

import tifffile
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

import sys
sys.path.append(str(Path.cwd().parent))
import conf

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

ANNOTATION_PATH = conf.ANNOTATION_PATH
CNN_INPUT_HEIGHT = conf.CNN_INPUT_HEIGHT
CNN_INPUT_WIDTH = conf.CNN_INPUT_WIDTH

class QuadtreeWSIPairDataset(Dataset):
    def __init__(self, annotation_path, input_height=344, input_width = 512):
        self.annotation_path = Path(annotation_path)
        self.input_height = input_height
        self.input_width = input_width

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            self.tile_jobs = json.load(f)

        # cache is per Dataset instance / per worker
        self._page_cache = {}

    def __len__(self):
        return len(self.tile_jobs)

    def _load_page_array(self, path, pyramid_page_idx):
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

    def _resize_gray_to_tensor(self, tile):
        if tile.dtype != np.uint8:
            tile = tile.astype(np.uint8)

        image = Image.fromarray(tile).convert("L")
        image = image.resize(
            (self.input_width, self.input_height),
            resample=Image.BILINEAR,
        )

        return TF.to_tensor(image)  # [1, H, W]


    def _resize_rgb_to_tensor(self, tile):
        if tile.dtype != np.uint8:
            tile = tile.astype(np.uint8)

        image = Image.fromarray(tile).convert("RGB")
        image = image.resize(
            (self.input_width, self.input_height),
            resample=Image.BILINEAR,
        )

        return TF.to_tensor(image)  # [3, H, W]
    
    def __getitem__(self, idx):
        job = self.tile_jobs[idx]

        fixed_page = self._load_page_array(
            path=job["fixed_path"],
            pyramid_page_idx=job["pyramid_page_idx"],
        )

        moving_page = self._load_page_array(
            path=job["moving_path"],
            pyramid_page_idx=job["pyramid_page_idx"],
        )

        fixed_tile = self._crop_tile(
            img=fixed_page,
            x_idx=job["x_idx"],
            y_idx=job["y_idx"],
            grid=job["grid"],
        )

        moving_tile = self._crop_tile(
            img=moving_page,
            x_idx=job["x_idx"],
            y_idx=job["y_idx"],
            grid=job["grid"],
        )

        fixed_tensor = self._resize_gray_to_tensor(fixed_tile)
        moving_tensor = self._resize_gray_to_tensor(moving_tile)

        fixed_vis = self._resize_rgb_to_tensor(fixed_tile)
        moving_vis = self._resize_rgb_to_tensor(moving_tile)

        transform = torch.tensor(
            job["transformation_matrix"],
            dtype=torch.float32,
        )

        registration_error = torch.tensor(
            job["registration_error"],
            dtype=torch.float32,
        )

        meta = {
            "pair_id": job["pair_id"],
            "source_image_id": job["source_image_id"],
            "target_image_id": job["target_image_id"],
            "crop_depth": job["crop_depth"],
            "grid": job["grid"],
            "x_idx": job["x_idx"],
            "y_idx": job["y_idx"],
            "pyramid_page_idx": job["pyramid_page_idx"],
            "tile_h": job["tile_h"],
            "tile_w": job["tile_w"],
        }

        return {
            "fixed": fixed_tensor,
            "moving": moving_tensor,
            "fixed_vis": fixed_vis,
            "moving_vis": moving_vis,
            "transform": transform,
            "registration_error": registration_error,
            "meta": meta,
        }
    


dataset = QuadtreeWSIPairDataset(
    annotation_path=ANNOTATION_PATH,
    input_height=CNN_INPUT_HEIGHT,
    input_width=CNN_INPUT_WIDTH
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,      # full deterministic traversal through the index
    num_workers=0,      # start with 0; later try 2
    pin_memory=True,
)


"""

print("dataset length:", len(dataset))
batch = next(iter(loader))

print(batch["fixed"].shape)
print(batch["moving"].shape)
print(batch["transform"].shape)
print(batch["registration_error"].shape)
print(batch["meta"])

"""