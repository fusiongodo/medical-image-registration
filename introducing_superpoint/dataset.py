"""
Matching HE/IHC tile pairs with HE pseudo-GT keypoints for SuperPoint retraining.
Loads pre-cropped PNGs from data/cropped_smooth/ and per-tile keypoints.json.

StainPairKeypointDataset[i] -> {
    "image_he":     [1, CNN_H, CNN_W] float32 in [0, 1]   (fixed / H&E)
    "image_ihc":    [1, CNN_H, CNN_W] float32 in [0, 1]   (moving / IHC, smooth-aligned)
    "gt_keypoints": [N, 3] float32 — (x, y, conf) in HE CNN pixels
    "meta":         dict with pair_id, depth (str "dN"), tile_id, tile_dir
}
make_loader(...) -> DataLoader; collate keeps gt_keypoints as list[Tensor].

split="train" | "val" | "all" — deterministic per (pair_id, depth) group.
Val counts: depth 4 → 25, depth 5 → 100, others → max(1, round(N * 0.1)).

preload=True loads all images and keypoints into RAM at init, eliminating
per-batch disk / NFS reads. Recommended when the dataset fits in memory.
"""
import sys
import json
import importlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

sys.path.append(str(Path(__file__).resolve().parent.parent))
import conf
importlib.reload(conf)

_DEFAULT_CROPPED_DIR = Path(conf.PROJECT_ROOT) / "data" / "cropped_smooth"

_VAL_COUNT_BY_DEPTH = {4: 25, 5: 100}
_DEFAULT_VAL_FRACTION = 0.1


def _val_count(depth_int: int, group_size: int) -> int:
    if depth_int in _VAL_COUNT_BY_DEPTH:
        return min(_VAL_COUNT_BY_DEPTH[depth_int], group_size)
    return max(1, round(group_size * _DEFAULT_VAL_FRACTION))


class StainPairKeypointDataset(Dataset):
    """
    cropped_dir: root of the pre-cropped tile tree.
                 Expected layout: {pair_id}/d{depth}/{tile}/he.png
                                                           ihc.png
                                                           keypoints.json
    Only tiles that have both he.png and keypoints.json are included.
    split: "train" | "val" | "all"
    """

    def __init__(
        self,
        cropped_dir=None,
        input_height=None,
        input_width=None,
        split: str = "all",
        preload: bool = False,
    ):
        self.cropped_dir  = Path(cropped_dir or _DEFAULT_CROPPED_DIR)
        self.input_height = input_height or conf.CNN_INPUT_HEIGHT
        self.input_width  = input_width  or conf.CNN_INPUT_WIDTH
        self.split        = split
        self.tile_dirs    = self._scan()
        self._cache: list | None = None
        if preload:
            self._preload()

    def _scan(self):
        groups: dict[tuple, list] = defaultdict(list)
        for pair_dir in sorted(self.cropped_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            for depth_dir in sorted(pair_dir.iterdir()):
                if not depth_dir.is_dir():
                    continue
                depth_str = depth_dir.name          # e.g. "d5"
                depth_int = int(depth_str.lstrip("d"))
                for tile_dir in sorted(depth_dir.iterdir()):
                    if not tile_dir.is_dir():
                        continue
                    if (
                        (tile_dir / "he.png").exists()
                        and (tile_dir / "ihc.png").exists()
                        and (tile_dir / "keypoints.json").exists()
                    ):
                        groups[(pair_dir.name, depth_int, depth_str)].append(tile_dir)

        if self.split == "all":
            return [t for group in groups.values() for t in group]

        tiles = []
        for (pair_id, depth_int, _depth_str), group in groups.items():
            n_val = _val_count(depth_int, len(group))
            val_set = set(group[:n_val])
            if self.split == "val":
                tiles.extend(group[:n_val])
            else:
                tiles.extend(t for t in group if t not in val_set)
        return tiles

    def _preload(self):
        print(f"preloading {len(self.tile_dirs)} tiles into RAM …", flush=True)
        self._cache = [self._load_item(td) for td in self.tile_dirs]
        print("preload complete", flush=True)

    def __len__(self):
        return len(self.tile_dirs)

    def _load_image(self, path):
        image = Image.open(path).convert("L")
        image = image.resize(
            (self.input_width, self.input_height),
            resample=Image.BILINEAR,
        )
        return TF.to_tensor(image)

    def _load_item(self, tile_dir: Path) -> dict:
        image_he  = self._load_image(tile_dir / "he.png")
        image_ihc = self._load_image(tile_dir / "ihc.png")

        with open(tile_dir / "keypoints.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        keypoints = raw if isinstance(raw, list) else raw.get("keypoints", [])
        gt_keypoints = torch.tensor(keypoints, dtype=torch.float32).reshape(-1, 3)

        meta = {
            "pair_id":  tile_dir.parent.parent.name,
            "depth":    tile_dir.parent.name,
            "tile_id":  tile_dir.name,
            "tile_dir": str(tile_dir),
        }

        return {
            "image_he":     image_he,
            "image_ihc":    image_ihc,
            "gt_keypoints": gt_keypoints,
            "meta":         meta,
        }

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_item(self.tile_dirs[idx])


def collate_pairs(batch):
    return {
        "image_he":     torch.stack([item["image_he"]  for item in batch]),
        "image_ihc":    torch.stack([item["image_ihc"] for item in batch]),
        "gt_keypoints": [item["gt_keypoints"] for item in batch],
        "meta":         [item["meta"]          for item in batch],
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
